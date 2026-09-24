import torch
import math
import torch.nn as nn
from torch_scatter import scatter

class BatchObject:
    """一个简单的对象，用来替代 torch_geometric 的 Batch，方便 Trainer 挂载属性"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
        return self


class SeqTrajectoryBalance:
    """
    OP-GFN Baseline 终极完全体 (支持 21维平铺动作 + 多目标软帕累托排名)。
    融合了经典的序列自回归采样与先进的 ICLR 2025 Global Rank 多目标惩罚机制。
    """
    def __init__(self, env_ctx, cfg, max_nodes=50):
        self.ctx = env_ctx
        self.global_cfg = cfg
        self.cfg = cfg.algo.tb
        self.max_len = max_nodes
        self.is_conditioned = False
        self.bootstrap_own_reward = getattr(self.cfg, 'bootstrap_own_reward', False)
        self.motif_usage_loss_weight = float(
            getattr(self.global_cfg.algo, "motif_usage_loss_weight", 0.0)
        )
        self.motif_diversity_loss_weight = float(
            getattr(self.global_cfg.algo, "motif_diversity_loss_weight", 0.0)
        )

    def create_training_data_from_graphs(self, graphs):
        """
        离线数据构造。

        第一版策略：
        仍然逐氨基酸构造离线轨迹，不使用 motif 动作。
        这样最稳，不会引入 motif 分解错误。

        但是为了兼容 motif 模型，需要记录：
        1. 每个状态对应的 motif_count
        2. 每条轨迹的 motif 使用信息
        """
        data = []

        for graph in graphs:
            seq = self.ctx.mol_to_graph(graph)
            seq = seq[:self.max_len]

            state = []
            motif_count = 0

            traj = []
            motif_counts = []
            bck_logprobs = []

            num_aa_actions = 0
            num_motif_actions = 0
            motif_action_ids = []
            motif_action_names = []

            for token in seq:
                action_idx = int(token)

                # 离线轨迹当前状态和动作
                traj.append((list(state), action_idx))
                motif_counts.append(motif_count)

                # 执行动作。这里 action_idx 一定是 0~19 的氨基酸动作
                new_state, new_motif_count, _ = self.ctx.apply_action(
                    state,
                    motif_count,
                    action_idx,
                )

                # 计算反向概率。
                # 对逐氨基酸离线轨迹来说，一般仍然是 0，
                # 但这里统一调用环境函数，保证逻辑一致。
                bck_lp = self.ctx.backward_logprob_after_action(
                    new_state,
                    new_motif_count,
                    action_idx,
                )
                bck_logprobs.append(bck_lp)

                state = new_state
                motif_count = new_motif_count
                num_aa_actions += 1

            # 末尾补 STOP
            stop_action = self.ctx.stop_action_idx
            traj.append((list(state), stop_action))
            motif_counts.append(motif_count)
            bck_logprobs.append(0.0)

            data.append({
                "traj": traj,
                "motif_counts": motif_counts,
                "bck_logprobs": torch.tensor(bck_logprobs, dtype=torch.float32),
                "result": list(state),
                "is_valid": len(state) >= getattr(self.ctx, "min_length", 5),

                # 下面这些是给后面日志和分析用的
                "motif_count": motif_count,
                "num_aa_actions": num_aa_actions,
                "num_motif_actions": num_motif_actions,
                "num_total_actions": len(traj),
                "motif_action_ids": motif_action_ids,
                "motif_action_names": motif_action_names,
            })

        return data

    def set_is_conditioned(self, is_cond: bool):
        self.is_conditioned = is_cond

    def create_training_data_from_own_samples(self, model, n: int, cond_info: torch.Tensor, random_action_prob: float,torch_generator=None,):
        """
        在线采样器：单层 motif 动作空间版本。

        每一步动作可以是：
        1. 单个氨基酸
        2. 一个 motif
        3. STOP

        状态仍然保存完整氨基酸索引序列。
        motif_count 单独维护。
        """
        dev = next(model.parameters()).device
        cond_info = cond_info.to(dev)

        # 初始化 N 个空序列状态
        states = [self.ctx.get_initial_state() for _ in range(n)]

        # 新增：每条轨迹当前已经使用过几个 motif
        motif_counts = [0 for _ in range(n)]

        trajs = []
        for _ in range(n):
            trajs.append({
                "traj": [],
                "motif_counts": [],
                "bck_logprobs": [],
                "result": None,

                # 日志/分析字段
                "motif_count": 0,
                "num_aa_actions": 0,
                "num_motif_actions": 0,
                "num_total_actions": 0,
                "motif_action_ids": [],
                "motif_action_names": [],
            })

        active_mask = torch.ones(n, dtype=torch.bool, device=dev)

        # 注意：这里仍然用 max_len + 1 作为最大动作步数上限。
        # 因为 motif 可以一次 append 多个氨基酸，所以实际通常会更早 STOP 或达到 max_len。
        for step in range(self.max_len + 1):
            active_indices = active_mask.nonzero().squeeze(-1)
            if len(active_indices) == 0:
                break

            current_states = [states[i] for i in active_indices.tolist()]
            current_motif_counts = [motif_counts[i] for i in active_indices.tolist()]

            # 关键变化：collate 现在要传 motif_counts
            batch = self.ctx.collate(current_states, current_motif_counts)
            batch["cond_info"] = cond_info[active_indices]

            for key in batch:
                batch[key] = batch[key].to(dev)

            with torch.no_grad():
                logits = model(batch)

                mask = batch["action_mask"].bool()

                # 防御：如果环境给了全 False 的 mask，强行放开所有动作
                mask_sum = mask.sum(dim=-1, keepdim=True)
                mask = torch.where(mask_sum == 0, torch.ones_like(mask), mask)

                logits = logits.masked_fill(~mask, -1e9)
                probs = torch.softmax(logits, dim=-1)

                # 保留你原来的 epsilon-greedy 探索逻辑
                if random_action_prob > 0:
                    uniform = mask.float() / mask.float().sum(dim=-1, keepdim=True)
                    probs = (1 - random_action_prob) * probs + random_action_prob * uniform

                probs = torch.clamp(probs, min=1e-10)
                probs = torch.nan_to_num(probs, nan=1e-10, posinf=1.0, neginf=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # actions = torch.multinomial(probs, 1).squeeze(-1)
                actions = torch.multinomial(probs, 1, generator=torch_generator,).squeeze(-1)

            # 更新状态和轨迹
            for idx, action_idx in enumerate(actions.tolist()):
                orig_i = active_indices[idx].item()
                action_idx = int(action_idx)

                old_state = list(states[orig_i])
                old_motif_count = int(motif_counts[orig_i])

                # 记录当前状态、动作、当前 motif_count
                trajs[orig_i]["traj"].append((old_state, action_idx))
                trajs[orig_i]["motif_counts"].append(old_motif_count)

                if self.ctx.is_stop_action(action_idx):
                    # STOP 的反向路径唯一
                    trajs[orig_i]["bck_logprobs"].append(0.0)
                    active_mask[orig_i] = False
                    trajs[orig_i]["result"] = list(states[orig_i])

                else:
                    # 关键变化：
                    # 不再 state.append(action_idx)
                    # 而是通过 ctx.apply_action 自动处理氨基酸动作或 motif 动作
                    new_state, new_motif_count, _ = self.ctx.apply_action(
                        old_state,
                        old_motif_count,
                        action_idx,
                    )

                    # 关键变化：
                    # 非 STOP 动作的 logP_B 不再全是 0，
                    # 而是均匀合法父状态反向概率。
                    bck_lp = self.ctx.backward_logprob_after_action(
                        new_state,
                        new_motif_count,
                        action_idx,
                    )
                    trajs[orig_i]["bck_logprobs"].append(bck_lp)

                    # 更新真实状态
                    states[orig_i] = new_state
                    motif_counts[orig_i] = new_motif_count

                    # 记录动作类型统计
                    if self.ctx.is_motif_action(action_idx):
                        trajs[orig_i]["num_motif_actions"] += 1
                        trajs[orig_i]["motif_action_ids"].append(action_idx)
                        trajs[orig_i]["motif_action_names"].append(
                            self.ctx.action_to_string(action_idx)
                        )
                    elif self.ctx.is_aa_action(action_idx):
                        trajs[orig_i]["num_aa_actions"] += 1

                trajs[orig_i]["num_total_actions"] += 1
                trajs[orig_i]["motif_count"] = motif_counts[orig_i]

        # 计算 logZ
        logZ_pred = model.logZ(cond_info)

        data = []
        for i in range(n):
            res_seq = trajs[i]["result"] if trajs[i]["result"] is not None else states[i]

            # 注意：这里先放 CPU，construct_batch 里再统一拼接，避免 offline/online 设备不一致
            b_logprobs = torch.tensor(
                trajs[i]["bck_logprobs"],
                dtype=torch.float32,
            )

            data.append({
                "traj": trajs[i]["traj"],
                "motif_counts": trajs[i]["motif_counts"],
                "bck_logprobs": b_logprobs,
                "result": res_seq,
                "logZ": logZ_pred[i].item(),
                "is_valid": len(res_seq) >= getattr(self.ctx, "min_length", 5),

                # metadata
                "motif_count": trajs[i]["motif_count"],
                "num_aa_actions": trajs[i]["num_aa_actions"],
                "num_motif_actions": trajs[i]["num_motif_actions"],
                "num_total_actions": trajs[i]["num_total_actions"],
                "motif_action_ids": trajs[i]["motif_action_ids"],
                "motif_action_names": trajs[i]["motif_action_names"],
            })

        return data

    def construct_batch(self, trajs, cond_info, log_rewards):
        """
        把采样到的多条轨迹打包成 batch。

        新增：
        1. 每个中间状态对应的 motif_count
        2. 每条轨迹的 motif 使用统计
        """
        all_states = []
        all_actions = []
        all_state_motif_counts = []
        traj_lens = []

        traj_motif_counts = []
        traj_num_aa_actions = []
        traj_num_motif_actions = []
        traj_num_total_actions = []

        for tj in trajs:
            traj_lens.append(len(tj["traj"]))

            state_motif_counts = tj.get(
                "motif_counts",
                [0 for _ in range(len(tj["traj"]))]
            )

            for step_idx, (state, action) in enumerate(tj["traj"]):
                all_states.append(state)
                all_actions.append(action)
                all_state_motif_counts.append(int(state_motif_counts[step_idx]))

            traj_motif_counts.append(int(tj.get("motif_count", 0)))
            traj_num_aa_actions.append(int(tj.get("num_aa_actions", 0)))
            traj_num_motif_actions.append(int(tj.get("num_motif_actions", 0)))
            traj_num_total_actions.append(int(tj.get("num_total_actions", len(tj["traj"]))))

        # 关键变化：collate 现在传入每个状态的 motif_count
        batch_dict = self.ctx.collate(all_states, all_state_motif_counts)

        batch_dict["actions"] = torch.tensor(all_actions, dtype=torch.long)
        batch_dict["traj_lens"] = torch.tensor(traj_lens, dtype=torch.long)

        # 注意统一放 CPU，后面 batch.to(device) 会整体搬到 GPU
        batch_dict["log_p_B"] = torch.cat(
            [tj["bck_logprobs"].detach().cpu() for tj in trajs],
            0,
        )

        batch_dict["log_rewards"] = log_rewards
        batch_dict["cond_info"] = cond_info
        batch_dict["is_valid"] = torch.tensor(
            [tj.get("is_valid", True) for tj in trajs]
        ).float()

        # trajectory-level metadata，主要用于训练日志和后续分析
        batch_dict["traj_motif_counts"] = torch.tensor(traj_motif_counts, dtype=torch.long)
        batch_dict["traj_num_aa_actions"] = torch.tensor(traj_num_aa_actions, dtype=torch.long)
        batch_dict["traj_num_motif_actions"] = torch.tensor(traj_num_motif_actions, dtype=torch.long)
        batch_dict["traj_num_total_actions"] = torch.tensor(traj_num_total_actions, dtype=torch.long)

        return BatchObject(**batch_dict)

    def compute_batch_losses(self, model, batch, num_bootstrap=0):
        """核心 Loss 计算逻辑 (融合了 ICLR 2025 全局排名)"""
        dev = batch.x.device
        num_trajs = len(batch.traj_lens)
        
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        
        # 修复 cond_info 维度对应问题
        model_inputs = batch.__dict__.copy()
        model_inputs["cond_info"] = batch.cond_info[batch_idx]
        
        # 1. 前向概率 log P_F
        logits = model(model_inputs) 
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        log_p_F = log_probs.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
        
        # 2. 回溯概率 log P_B
        log_p_B = batch.log_p_B.to(dev)

        # 3. 累加到轨迹级别
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        # 4. 获取配分函数和裁剪奖励
        log_Z = model.logZ(batch.cond_info)[:, 0]
        clip_log_R = torch.maximum(
            batch.log_rewards, torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=dev)
        ).float()

        # 5. 目标分支计算
        if self.cfg.do_ordering:
            pred_log_R = log_Z + traj_log_p_F - traj_log_p_B
            
            # ==========================================
            # 🌟 核心保留：路线 A / B 动态切换引擎
            # ==========================================
            use_global_rank = getattr(self, 'use_global_rank', False)
            
            if use_global_rank:
                # 路线 B: ICLR 2025 Global Rank (多目标帕累托前沿软标签)
                soft_targets = self.get_soft_pareto_labels(batch.log_rewards, temperature=2.0)
                traj_losses = torch.nn.CrossEntropyLoss()(pred_log_R.unsqueeze(0), soft_targets.unsqueeze(0))
            else:
                # 路线 A: 原始硬截断
                traj_losses = self.ordering_loss_epoch(batch.is_pareto, pred_log_R)
                
            loss = traj_losses
        else:
            numerator = log_Z + traj_log_p_F
            denominator = clip_log_R + traj_log_p_B
            traj_losses = (numerator - denominator).pow(2)
            loss = traj_losses.mean()

        # ==========================================================
        # Offline Stop Loss:
        # 只监督离线短肽轨迹的最后一步，让模型在短肽终点更愿意 STOP。
        # 不修改 amp/hemo reward，也不影响 Pareto 分数。
        # ==========================================================
        offline_stop_loss_weight = getattr(self, "offline_stop_loss_weight", 0.0)

        if (
            offline_stop_loss_weight > 0
            and hasattr(batch, "num_offline")
            and batch.num_offline > 0
        ):
            # 每条轨迹摊平成状态-动作对后，找到每条轨迹最后一个动作的位置
            traj_end_idx = torch.cumsum(batch.traj_lens, dim=0) - 1

            # SamplingIterator 里离线轨迹排在 batch 最前面
            offline_end_idx = traj_end_idx[:batch.num_offline]

            # 离线短肽终点状态下，STOP 动作的 log probability
            stop_log_probs = log_probs[offline_end_idx, self.ctx.stop_action_idx]

            # 最大化 P(STOP | 离线短肽终点)
            stop_loss = -stop_log_probs.mean()

            loss = loss + offline_stop_loss_weight * stop_loss

            offline_stop_prob = stop_log_probs.exp().mean()
            offline_end_is_stop = (
                batch.actions[offline_end_idx] == self.ctx.stop_action_idx
            ).float().mean()
        else:
            stop_loss = torch.tensor(0.0, device=dev)
            offline_stop_prob = torch.tensor(0.0, device=dev)
            offline_end_is_stop = torch.tensor(0.0, device=dev)

        # ==========================================================
        # Online Stop Loss:
        # 不修改 MIC/HEMO/MIC-TOX reward。
        # 对 online 轨迹中长度 >= online_stop_len 的中间状态，
        # 鼓励模型提高 STOP 动作概率。
        #
        # 注意：
        # 这不是长度 reward；它只作用于策略的 STOP 行为。
        # ==========================================================
        online_stop_loss_weight = float(
            getattr(self, "online_stop_loss_weight", 0.0)
        )

        online_stop_len = int(
            getattr(self, "online_stop_len", 20)
        )

        online_stop_loss = torch.tensor(0.0, device=dev)
        online_stop_prob = torch.tensor(0.0, device=dev)
        online_stop_state_frac = torch.tensor(0.0, device=dev)

        if (
            online_stop_loss_weight > 0
            and hasattr(batch, "num_offline")
            and int(getattr(batch, "num_online", 0)) > 0
        ):
            num_offline = int(getattr(batch, "num_offline", 0))

            # trajectory 级别：哪些轨迹是 online
            traj_is_online = torch.arange(num_trajs, device=dev) >= num_offline

            # state 级别：每个展开状态属于哪条轨迹
            state_is_online = traj_is_online[batch_idx]

            # 每个展开状态当前的序列长度
            state_lengths = batch.lengths.to(dev)

            # 只在 STOP 合法的状态上监督
            action_mask = model_inputs["action_mask"].bool().to(dev)
            stop_is_legal = action_mask[:, self.ctx.stop_action_idx]

            online_stop_states = (
                state_is_online
                & stop_is_legal
                & (state_lengths >= online_stop_len)
            )

            if online_stop_states.any():
                online_stop_log_probs = log_probs[
                    online_stop_states,
                    self.ctx.stop_action_idx,
                ]

                online_stop_loss = -online_stop_log_probs.mean()
                loss = loss + online_stop_loss_weight * online_stop_loss

                online_stop_prob = online_stop_log_probs.exp().mean()
                online_stop_state_frac = online_stop_states.float().mean()

        motif_usage_loss_weight = float(getattr(self, "motif_usage_loss_weight", 0.0))
        motif_diversity_loss_weight = float(getattr(self, "motif_diversity_loss_weight", 0.0))

        motif_usage_penalty = torch.tensor(0.0, device=dev)
        motif_usage_loss = torch.tensor(0.0, device=dev)
        motif_diversity_score = torch.tensor(0.0, device=dev)
        motif_diversity_loss = torch.tensor(0.0, device=dev)

        motif_start = getattr(self.ctx, "motif_action_start", self.ctx.vocab_size)
        motif_end = getattr(self.ctx, "motif_action_end", self.ctx.stop_action_idx)

        if motif_end > motif_start:
            num_offline = int(getattr(batch, "num_offline", 0))
            traj_is_online = torch.arange(num_trajs, device=dev) >= num_offline
            state_is_online = traj_is_online[batch_idx]

            action_mask = model_inputs["action_mask"].bool()
            motif_mask = action_mask[:, motif_start:motif_end]

            motif_probs = probs[:, motif_start:motif_end] * motif_mask.float()
            state_motif_mass = motif_probs.sum(dim=1)

            online_state_motif_mass = torch.where(
                state_is_online,
                state_motif_mass,
                torch.zeros_like(state_motif_mass),
            )

            expected_motif_count = scatter(
                online_state_motif_mass,
                batch_idx,
                dim=0,
                dim_size=num_trajs,
                reduce="sum",
            )

            online_expected_motif_count = expected_motif_count[traj_is_online]

            if online_expected_motif_count.numel() > 0:
                motif_usage_penalty = online_expected_motif_count.mean()

            if motif_usage_loss_weight > 0:
                motif_usage_loss = motif_usage_loss_weight * motif_usage_penalty
                loss = loss + motif_usage_loss

            if motif_diversity_loss_weight > 0:
                eligible_states = state_is_online & motif_mask.any(dim=1)

                if eligible_states.any():
                    motif_mass_by_id = motif_probs[eligible_states].sum(dim=0)
                    legal_motif_ids = motif_mask[eligible_states].any(dim=0)
                    motif_mass_by_id = motif_mass_by_id[legal_motif_ids]

                    if motif_mass_by_id.numel() > 1:
                        motif_dist = motif_mass_by_id / motif_mass_by_id.sum().clamp_min(1e-8)
                        motif_entropy = -(
                            motif_dist * motif_dist.clamp_min(1e-8).log()
                        ).sum()

                        max_entropy = math.log(float(motif_dist.numel()))
                        motif_diversity_score = motif_entropy / max_entropy

                        motif_diversity_loss = -motif_diversity_loss_weight * motif_diversity_score
                        loss = loss + motif_diversity_loss

        info = {
            "loss": loss.item(),
            "logZ": log_Z.mean().item(),
            "invalid_trajectories": (1 - batch.is_valid).mean().item(),
            "num_offline": float(getattr(batch, "num_offline", -1)),
            "num_online": float(getattr(batch, "num_online", -1)),
            "offline_stop_loss": stop_loss.item(),
            "offline_stop_prob": offline_stop_prob.item(),
            "offline_end_is_stop": offline_end_is_stop.item(),
            "offline_stop_loss_weight": float(offline_stop_loss_weight),
            "motif_usage_penalty": motif_usage_penalty.item(),
            "motif_usage_loss": motif_usage_loss.item(),
            "motif_usage_loss_weight": float(motif_usage_loss_weight),
            "motif_diversity_score": motif_diversity_score.item(),
            "motif_diversity_loss": motif_diversity_loss.item(),
            "motif_diversity_loss_weight": float(motif_diversity_loss_weight),
            "online_stop_loss": online_stop_loss.item(),
            "online_stop_prob": online_stop_prob.item(),
            "online_stop_state_frac": online_stop_state_frac.item(),
            "online_stop_loss_weight": float(online_stop_loss_weight),
            "online_stop_len": float(online_stop_len),
            "pareto_len_limit": float(getattr(self, "pareto_len_limit", 30)),
            "replay_len_limit": float(getattr(self, "replay_len_limit", 30)),
        }

        # 新增：观察 motif 动作是否被使用
        if hasattr(batch, "traj_motif_counts"):
            info["motif_count_mean"] = batch.traj_motif_counts.float().mean().item()
            info["motif_action_mean"] = batch.traj_num_motif_actions.float().mean().item()
            info["aa_action_mean"] = batch.traj_num_aa_actions.float().mean().item()
            info["total_action_mean"] = batch.traj_num_total_actions.float().mean().item()
        return loss, info

    def ordering_loss_epoch(self, zero_one, pred_log_rewards):
        zero_one_float = zero_one.float()
        s = torch.sum(zero_one_float)
        if s <= 0:
            return (pred_log_rewards * 0.0).mean()
        return torch.nn.CrossEntropyLoss()(pred_log_rewards.unsqueeze(0), (zero_one_float / s).unsqueeze(0))
    
    def get_soft_pareto_labels(self, log_rewards, temperature=2.0):
        """
        [ICLR 2025 路线 B]: Global Rank 多级 Pareto 前沿软标签算法
        保留自 H-OP-GFN，完美处理双/三目标的多维张量帕累托分层！
        """
        N = log_rewards.shape[0]
        dev = log_rewards.device
        ranks = torch.zeros(N, device=dev)
        
        if log_rewards.dim() == 1:
            unique_vals, inverse_indices = torch.unique(log_rewards, sorted=True, return_inverse=True)
            inverted_ranks = inverse_indices.float()
            return torch.softmax(inverted_ranks * temperature, dim=0)

        active_mask = torch.ones(N, dtype=torch.bool, device=dev)
        current_rank = 0
        max_loops = N + 5  
        
        while active_mask.any() and current_rank < max_loops:
            active_idx = active_mask.nonzero().squeeze(-1)
            if active_idx.dim() == 0:
                active_idx = active_idx.unsqueeze(0)
                
            active_rewards = log_rewards[active_idx]
            
            diff = active_rewards.unsqueeze(1) - active_rewards.unsqueeze(0)
            geq = (diff >= 0).all(dim=-1)
            gt = (diff > 0).any(dim=-1)
            dominates = geq & gt  
            
            is_dominated = dominates.any(dim=0)  
            front_mask = ~is_dominated
            
            if not front_mask.any():
                front_mask[0] = True
                
            front_idx = active_idx[front_mask]
            ranks[front_idx] = float(current_rank)
            
            active_mask[front_idx] = False
            current_rank += 1
            
        inverted_ranks = (current_rank - 1) - ranks
        return torch.softmax(inverted_ranks * temperature, dim=0)

    def ordering_loss(self, log_rewards: torch.Tensor, pred_log_rewards: torch.Tensor):
        assert log_rewards.shape == pred_log_rewards.shape
        log_rewards = log_rewards[:, None]
        pred_log_rewards = pred_log_rewards[:, None]
        
        shuffling_indices = torch.randperm(log_rewards.shape[0])
        log_rewards = torch.cat([log_rewards, log_rewards[shuffling_indices]], dim=1)
        pred_log_rewards = torch.cat([pred_log_rewards, pred_log_rewards[shuffling_indices]], dim=1)
        
        tmp = ((log_rewards[:, 0] > log_rewards[:, 1]).float() + (log_rewards[:, 0] > log_rewards[:, 1] - 1e-6).float()) / 2.0
        log_rewards_target = torch.cat([tmp[:, None], (1 - tmp)[:, None]], dim=1)

        return torch.nn.CrossEntropyLoss()(pred_log_rewards, log_rewards_target)