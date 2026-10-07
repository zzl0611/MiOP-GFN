import os
import sqlite3
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, IterableDataset
from botorch.utils.multi_objective import pareto

from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.envs.graph_building_env import GraphActionCategorical


class SamplingIterator(IterableDataset):
    """This class allows us to parallelise and train faster.

    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound.

    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        ctx,
        algo,
        task,
        device,
        batch_size: int = 1,
        illegal_action_logreward: float = -50,
        ratio: float = 0.5,
        stream: bool = True,
        replay_buffer: ReplayBuffer = None,
        log_dir: str = None,
        sample_cond_info: bool = True,
        random_action_prob: float = 0.0,
        hindsight_ratio: float = 0.0,
        init_train_iter: int = 0,
        use_pareto: bool = False,
    ):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        ctx:
            The context for the environment, e.g. a MolBuildingEnvContext instance
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: GFNTask
            A Task instance, e.g. a MakeRingsTask instance
        device: torch.device
            The device the model is on
        replay_buffer: ReplayBuffer
            The replay buffer for training on past data
        batch_size: int
            The number of trajectories, each trajectory will be comprised of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        illegal_action_logreward: float
            The logreward for invalid trajectories
        ratio: float
            The ratio of offline trajectories in the batch.
        stream: bool
            If True, data is sampled iid for every batch. Otherwise, this is a normal in-order
            dataset iterator.
        log_dir: str
            If not None, logs each SamplingIterator worker's generated molecules to that file.
        sample_cond_info: bool
            If True (default), then the dataset is a dataset of points used in offline training.
            If False, then the dataset is a dataset of preferences (e.g. used to validate the model)
        random_action_prob: float
            The probability of taking a random action, passed to the graph sampler
        init_train_iter: int
            The initial training iteration, incremented and passed to task.sample_conditional_information
        """
        self.data = dataset
        self.model = model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.illegal_action_logreward = illegal_action_logreward
        self.offline_batch_size = int(np.ceil(self.batch_size * ratio))
        self.online_batch_size = int(np.floor(self.batch_size * (1 - ratio)))
        self.ratio = ratio
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.device = device
        self.stream = stream
        self.sample_online_once = True  # TODO: deprecate this, disallow len(data) == 0 entirely
        self.sample_cond_info = sample_cond_info
        self.random_action_prob = random_action_prob
        self.hindsight_ratio = hindsight_ratio
        self.train_it = init_train_iter
        self.do_validate_batch = False  # Turn this on for debugging
        self.log_molecule_smis = not hasattr(self.ctx, "not_a_molecule_env")  # TODO: make this a proper flag
        self.use_pareto = use_pareto

        # Slightly weird semantics, but if we're sampling x given some fixed cond info (data)
        # then "offline" now refers to cond info and online to x, so no duplication and we don't end
        # up with 2*batch_size accidentally
        if not sample_cond_info:
            self.offline_batch_size = self.online_batch_size = self.batch_size

        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as where the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        self.log_dir = log_dir
        self.log = SQLiteLog()
        self.log_hooks: List[Callable] = []

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def _idx_iterator(self):
        RDLogger.DisableLog("rdApp.*")
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            while True:
                if len(self.data) < self.offline_batch_size:
                    yield self.rng.integers(0, 0, 0)
                else:
                    yield self.rng.integers(0, len(self.data), self.offline_batch_size)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)
            if n == 0:
                yield np.arange(0, 0)
                return
            assert (
                self.offline_batch_size > 0
            ), "offline_batch_size must be > 0 if not streaming and len(data) > 0 (have you set ratio=0?)"
            if worker_info is None:  # no multi-processing
                start, end, wid = 0, n, -1
            else:  # split the data into chunks (per-worker)
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))
            bs = self.offline_batch_size
            if end - start <= bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        if self.stream:
            return int(1e6)
        if len(self.data) == 0 and self.sample_online_once:
            return 1
        return len(self.data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        # self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        base_seed = int(getattr(self.task.cfg, "seed", 0))
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(base_seed + self._wid)
        self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            self.log.connect(self.log_path)

        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [0, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.

            if self.sample_cond_info:
                num_online = self.online_batch_size
                cond_info = self.task.sample_conditional_information(
                    num_offline + self.online_batch_size, self.train_it
                )

                # Sample some dataset data
                mols, flat_rewards = map(list, zip(*[self.data[i] for i in idcs])) if len(idcs) else ([], [])
                flat_rewards = (
                    list(self.task.flat_reward_transform(torch.stack(flat_rewards))) if len(flat_rewards) else []
                )
                graphs = [self.ctx.mol_to_graph(m) for m in mols]
                trajs = self.algo.create_training_data_from_graphs(graphs)

            else:  # If we're not sampling the conditionals, then the idcs refer to listed preferences
                num_online = num_offline
                num_offline = 0
                cond_info = self.task.encode_conditional_information(
                    steer_info=torch.stack([self.data[i] for i in idcs])
                )
                trajs, flat_rewards = [], []

            # Sample some on-policy data
            is_valid = torch.ones(num_offline + num_online).bool()
            if num_online > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(
                        self.model,
                        num_online,
                        cond_info["encoding"][num_offline:],
                        random_action_prob=self.random_action_prob,
                    )
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_logreward
                    pred_reward = [i["reward_pred"].cpu().item() for i in trajs[num_offline:]]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.tensor(
                        [i + num_offline for i in range(num_online) if trajs[i + num_offline]["is_valid"]]
                    ).long()
                    # fetch the valid trajectories endpoints
                    mols = [self.ctx.graph_to_mol(trajs[i]["result"]) for i in valid_idcs]
                    # ask the task to compute their reward
                    online_flat_rew, m_is_valid = self.task.compute_flat_rewards(mols)
                    assert (
                        online_flat_rew.ndim == 2
                    ), "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    valid_mols = [m for m, v in zip(mols, m_is_valid) if v]
                    pred_reward = torch.zeros((num_online, online_flat_rew.shape[1]))
                    pred_reward[valid_idcs - num_offline] = online_flat_rew
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(num_online):
                        trajs[num_offline + i]["is_valid"] = is_valid[num_offline + i].item()
                    if self.log_molecule_smis:
                        # for i, m in zip(valid_idcs, valid_mols):
                        #     trajs[i]["smi"] = Chem.MolToSmiles(m)
                        for i, m in zip(valid_idcs, mols):
                            # 终极兼容补丁：如果是分子，就正常转换；如果是我们的序列，就直接变成字符串！
                            try:
                                trajs[i]["smi"] = Chem.MolToSmiles(m)
                            except Exception:
                                trajs[i]["smi"] = str(m)

            # Compute scalar rewards from conditional information & flat rewards
            flat_rewards = torch.stack(flat_rewards)
            log_rewards = self.task.cond_info_to_logreward(cond_info, flat_rewards)
            log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward

            # Computes some metrics
            extra_info = {}
            if not self.sample_cond_info:
                # If we're using a dataset of preferences, the user may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
            #  note: we convert back into natural rewards for logging purposes
            #  (allows to take averages and plot in objective space)
            #  TODO: implement that per-task (in case they don't apply the same beta and log transformations)
            rewards = torch.exp(log_rewards / cond_info["beta"])
            if num_online > 0 and self.log_dir is not None:
                self.log_generated(
                    deepcopy(trajs[num_offline:]),
                    deepcopy(rewards[num_offline:]),
                    deepcopy(flat_rewards[num_offline:]),
                    {k: v[num_offline:] for k, v in deepcopy(cond_info).items()},
                )
            if num_online > 0:
                for hook in self.log_hooks:
                    extra_info.update(
                        hook(
                            deepcopy(trajs[num_offline:]),
                            deepcopy(rewards[num_offline:]),
                            deepcopy(flat_rewards[num_offline:]),
                            {k: v[num_offline:] for k, v in deepcopy(cond_info).items()},
                        )
                    )

            if self.replay_buffer is not None:
                # cond_info is a dict, so we need to convert it to a list of dicts
                cond_info = [
                    {k: v[i] for k, v in cond_info.items()}
                    for i in range(num_offline + num_online)
                ]

                # ==========================================================
                # Replay length constraint:
                # 长度 > replay_len_limit 的 online 序列不放进 replay buffer。
                # 不修改 reward，只限制 replay 记忆。
                # ==========================================================
                replay_len_limit = int(getattr(self.algo, "replay_len_limit", 30))

                num_replay_pushed = 0
                num_replay_skipped_long = 0
                num_replay_skipped_invalid = 0

                for i in range(num_offline, len(trajs)):
                    seq_len = len(trajs[i].get("result", []))
                    valid_flag = bool(is_valid[i])

                    if not valid_flag:
                        num_replay_skipped_invalid += 1
                        continue

                    if seq_len > replay_len_limit:
                        num_replay_skipped_long += 1
                        continue

                    trajs[i]["replay_eligible"] = True
                    trajs[i]["seq_len"] = seq_len

                    self.replay_buffer.push(
                        deepcopy(trajs[i]),
                        deepcopy(log_rewards[i]),
                        deepcopy(flat_rewards[i]),
                        deepcopy(cond_info[i]),
                        deepcopy(is_valid[i]),
                    )

                    num_replay_pushed += 1

                extra_info["replay_pushed"] = float(num_replay_pushed)
                extra_info["replay_skipped_long"] = float(num_replay_skipped_long)
                extra_info["replay_skipped_invalid"] = float(num_replay_skipped_invalid)

                # ==========================================================
                # Replay sample safety:
                # replay buffer 为空时，保留当前 online 样本，不做 replay 替换。
                # ==========================================================
                if len(self.replay_buffer) > 0:
                    pareto_indices = getattr(self.replay_buffer, "pareto_indices", [])

                    if self.use_pareto and len(pareto_indices) > 0:
                        num_pareto = int(num_online * 0.1)
                    else:
                        num_pareto = 0

                    replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid = self.replay_buffer.sample(
                        num_online - num_pareto
                    )

                    if num_pareto:
                        trajs[num_offline:-num_pareto] = replay_trajs
                        log_rewards[num_offline:-num_pareto] = replay_logr
                        flat_rewards[num_offline:-num_pareto] = replay_fr
                        cond_info[num_offline:-num_pareto] = replay_condinfo
                        is_valid[num_offline:-num_pareto] = replay_valid

                        replay_trajs_p, replay_logr_p, replay_fr_p, replay_condinfo_p, replay_valid_p = self.replay_buffer.sample_pareto(
                            num_pareto
                        )

                        trajs[-num_pareto:] = replay_trajs_p
                        log_rewards[-num_pareto:] = replay_logr_p
                        flat_rewards[-num_pareto:] = replay_fr_p
                        cond_info[-num_pareto:] = replay_condinfo_p
                        is_valid[-num_pareto:] = replay_valid_p
                    else:
                        trajs[num_offline:] = replay_trajs
                        log_rewards[num_offline:] = replay_logr
                        flat_rewards[num_offline:] = replay_fr
                        cond_info[num_offline:] = replay_condinfo
                        is_valid[num_offline:] = replay_valid

                    extra_info["replay_replaced_online"] = 1.0
                    extra_info["replay_num_pareto"] = float(num_pareto)
                else:
                    extra_info["replay_replaced_online"] = 0.0
                    extra_info["replay_num_pareto"] = 0.0

                # # ==========================================================
                # # Replay sampling:
                # #
                # # 1. replay buffer 未达到 warmup 前：
                # #    不替换 fresh online trajectories。
                # #
                # # 2. OP-GFN ordering 模式 (use_pareto=True)：
                # #    warmup 后，online replay 100% 从当前 Pareto archive 采样。
                # #
                # # 3. 非 ordering 模式：
                # #    保留普通 replay 行为。
                # # ==========================================================
                # replay_warmup = int(getattr(self.replay_buffer, "warmup", 0))

                # if len(self.replay_buffer) >= replay_warmup:

                #     pareto_indices = getattr(
                #         self.replay_buffer,
                #         "pareto_indices",
                #         [],
                #     )

                #     # ------------------------------------------------------
                #     # OP-GFN ordering:
                #     # 100% Pareto replay
                #     # ------------------------------------------------------
                #     if self.use_pareto:

                #         if len(pareto_indices) > 0:

                #             (
                #                 replay_trajs,
                #                 replay_logr,
                #                 replay_fr,
                #                 replay_condinfo,
                #                 replay_valid,
                #             ) = self.replay_buffer.sample_pareto(num_online)

                #             trajs[num_offline:] = replay_trajs
                #             log_rewards[num_offline:] = replay_logr
                #             flat_rewards[num_offline:] = replay_fr
                #             cond_info[num_offline:] = replay_condinfo
                #             is_valid[num_offline:] = replay_valid

                #             extra_info["replay_replaced_online"] = 1.0
                #             extra_info["replay_num_pareto"] = float(num_online)

                #         else:
                #             # 理论上 warmup 后通常都会有 Pareto 点。
                #             # 如果异常为空，则保留当前 fresh online，
                #             # 不退化成普通 replay。
                #             extra_info["replay_replaced_online"] = 0.0
                #             extra_info["replay_num_pareto"] = 0.0

                #     # ------------------------------------------------------
                #     # 非 ordering 模式：
                #     # 保留普通 replay
                #     # ------------------------------------------------------
                #     else:

                #         (
                #             replay_trajs,
                #             replay_logr,
                #             replay_fr,
                #             replay_condinfo,
                #             replay_valid,
                #         ) = self.replay_buffer.sample(num_online)

                #         trajs[num_offline:] = replay_trajs
                #         log_rewards[num_offline:] = replay_logr
                #         flat_rewards[num_offline:] = replay_fr
                #         cond_info[num_offline:] = replay_condinfo
                #         is_valid[num_offline:] = replay_valid

                #         extra_info["replay_replaced_online"] = 1.0
                #         extra_info["replay_num_pareto"] = 0.0

                # else:
                #     # ------------------------------------------------------
                #     # warmup 阶段：
                #     # 当前生成的 fresh online trajectories 直接参与训练
                #     # ------------------------------------------------------
                #     extra_info["replay_replaced_online"] = 0.0
                #     extra_info["replay_num_pareto"] = 0.0


                # extra_info["replay_buffer_size"] = float(len(self.replay_buffer))
                # extra_info["replay_warmup"] = float(replay_warmup)
                # extra_info["pareto_archive_size"] = float(
                #     len(getattr(self.replay_buffer, "pareto_indices", []))
                # )

                # 无论有没有替换 online 样本，都要把 cond_info 转回 dict
                cond_info = {
                    k: torch.stack([d[k] for d in cond_info])
                    for k in cond_info[0]
                }

            if self.hindsight_ratio > 0.0:
                # Relabels some of the online trajectories with hindsight
                assert hasattr(
                    self.task, "relabel_condinfo_and_logrewards"
                ), "Hindsight requires the task to implement relabel_condinfo_and_logrewards"
                # samples indexes of trajectories without repeats
                hindsight_idxs = torch.randperm(num_online)[: int(num_online * self.hindsight_ratio)] + num_offline
                cond_info, log_rewards = self.task.relabel_condinfo_and_logrewards(
                    cond_info, log_rewards, flat_rewards, hindsight_idxs
                )
                log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward

            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], log_rewards)
            batch.num_offline = num_offline
            batch.num_online = num_online
            batch.flat_rewards = flat_rewards
            batch.preferences = cond_info.get("preferences", None)
            batch.focus_dir = cond_info.get("focus_dir", None)
            batch.extra_info = extra_info
            # ==========================================================
            # Constrained Pareto for training:
            # 长度 > pareto_len_limit 的序列不作为训练 batch 的 Pareto 正样本。
            # 不修改 MIC/HEMO/MIC-TOX reward。
            # ==========================================================
            pareto_len_limit = int(getattr(self.algo, "pareto_len_limit", 30))

            seq_lens = torch.tensor(
                [len(t.get("result", [])) for t in trajs],
                dtype=torch.long,
                device=flat_rewards.device,
            )

            pareto_eligible = (
                is_valid.to(flat_rewards.device).bool()
                & (seq_lens <= pareto_len_limit)
            )

            is_pareto = torch.zeros(
                flat_rewards.shape[0],
                dtype=torch.bool,
                device=flat_rewards.device,
            )

            if pareto_eligible.any():
                eligible_idx = pareto_eligible.nonzero(as_tuple=False).squeeze(-1)

                eligible_pareto = pareto.is_non_dominated(
                    flat_rewards[eligible_idx],
                    deduplicate=False,
                ).to(flat_rewards.device)

                is_pareto[eligible_idx] = eligible_pareto

            batch.seq_lens = seq_lens
            batch.pareto_eligible = pareto_eligible.float()
            batch.is_pareto = is_pareto.float()

            # 防御检查：训练 batch 里，长度 > limit 的样本绝不应该被标为 Pareto
            assert not ((seq_lens > pareto_len_limit) & is_pareto).any(), \
                "Found Pareto training samples with length > pareto_len_limit!"

            extra_info["pareto_eligible_frac"] = pareto_eligible.float().mean().item()
            extra_info["seq_len_mean"] = seq_lens.float().mean().item()
            extra_info["seq_len_max"] = seq_lens.float().max().item()
            extra_info["over_pareto_len_frac"] = (seq_lens > pareto_len_limit).float().mean().item()
            extra_info["pareto_len_limit"] = float(pareto_len_limit)
            
            # TODO: we could very well just pass the cond_info dict to construct_batch above,
            # and the algo can decide what it wants to put in the batch object

            # Only activate for debugging your environment or dataset (e.g. the dataset could be
            # generating trajectories with illegal actions)
            if self.do_validate_batch:
                self.validate_batch(batch, trajs)

            self.train_it += worker_info.num_workers if worker_info is not None else 1
            yield batch

    def validate_batch(self, batch, trajs):
        for actions, atypes in [(batch.actions, self.ctx.action_type_order)] + (
            [(batch.bck_actions, self.ctx.bck_action_type_order)]
            if hasattr(batch, "bck_actions") and hasattr(self.ctx, "bck_action_type_order")
            else []
        ):
            mask_cat = GraphActionCategorical(
                batch,
                [self.model._action_type_to_mask(t, batch) for t in atypes],
                [self.model._action_type_to_key[t] for t in atypes],
                [None for _ in atypes],
            )
            masked_action_is_used = 1 - mask_cat.log_prob(actions, logprobs=mask_cat.logits)
            num_trajs = len(trajs)
            batch_idx = torch.arange(num_trajs, device=batch.x.device).repeat_interleave(batch.traj_lens)
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            if masked_action_is_used.sum() != 0:
                invalid_idx = masked_action_is_used.argmax().item()
                traj_idx = batch_idx[invalid_idx].item()
                timestep = invalid_idx - first_graph_idx[traj_idx].item()
                raise ValueError("Found an action that was masked out", trajs[traj_idx]["traj"][timestep])

    def log_generated(self, trajs, rewards, flat_rewards, cond_info):
        if self.log_molecule_smis:
            # mols = [
            #     Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]["result"])) if trajs[i]["is_valid"] else ""
            #     for i in range(len(trajs))
            # ]
            # 终极数据库兼容补丁
            mols = []
            for i in range(len(trajs)):
                if not trajs[i].get("is_valid", False):
                    mols.append("")
                else:
                    obj = self.ctx.graph_to_mol(trajs[i]["result"])
                    try:
                        # 如果是原始的化学图网络，正常转 SMILES
                        mols.append(Chem.MolToSmiles(obj))
                    except Exception:
                    # 如果是咱们的氨基酸序列，直接转成字符串存进数据库！
                        mols.append(str(obj))
        else:
            mols = [nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(t["result"], None, "v") for t in trajs]

        flat_rewards = flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(mols), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(mols), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]


        seq_strings = []
        seq_lengths = []

        for t in trajs:
            result = t.get("result", [])

            if hasattr(self.ctx, "idx_to_seq_str"):
                seq_str = self.ctx.idx_to_seq_str(result)
            else:
                seq_str = str(result)

            seq_strings.append(seq_str)
            seq_lengths.append(len(seq_str))

        motif_counts = [int(t.get("motif_count", 0)) for t in trajs]
        num_aa_actions = [int(t.get("num_aa_actions", 0)) for t in trajs]
        num_motif_actions = [int(t.get("num_motif_actions", 0)) for t in trajs]
        num_total_actions = [int(t.get("num_total_actions", 0)) for t in trajs]

        motif_action_ids = [
            ",".join(map(str, t.get("motif_action_ids", [])))
            for t in trajs
        ]

        motif_action_names = [
            ";".join(t.get("motif_action_names", []))
            for t in trajs
        ]

        data = [
            [mols[i], seq_strings[i], seq_lengths[i], rewards[i]]
            + flat_rewards[i]
            + preferences[i]
            + focus_dir[i]
            + [
                motif_counts[i],
                num_aa_actions[i],
                num_motif_actions[i],
                num_total_actions[i],
                motif_action_ids[i],
                motif_action_names[i],
            ]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]

        data_labels = (
            ["smi", "seq", "seq_len", "r"]
            + [f"fr_{i}" for i in range(len(flat_rewards[0]))]
            + [f"pref_{i}" for i in range(len(preferences[0]))]
            + [f"focus_{i}" for i in range(len(focus_dir[0]))]
            + [
                "motif_count",
                "num_aa_actions",
                "num_motif_actions",
                "num_total_actions",
                "motif_action_ids",
                "motif_action_names",
            ]
            + [f"ci_{k}" for k in logged_keys]
        )

        self.log.insert_many(data, data_labels)


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(f"{name} {type_map[t]}" for t, name in zip(types, names))
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert all([type(x) is str or not isinstance(x, Iterable) for x in rows[0]]), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()
