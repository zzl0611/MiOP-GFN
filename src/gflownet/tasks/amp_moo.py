import os
# 与正式训练命令一致：外部未指定显卡时默认只暴露物理 GPU 0。
# 必须在导入 torch 前设置；外部 CUDA_VISIBLE_DEVICES 仍具有更高优先级。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import random
import numpy as np
import shutil
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor

# GFlowNet 底层引擎导入
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.tasks.seh_frag_moo import SEHMOOTask
from gflownet.trainer import FlatRewards

from gflownet.envs.seq_env import SeqBuildingEnvContext
from gflownet.models.seq_model import SeqTransformerGFN
from gflownet.algo.seq_trajectory_balance import SeqTrajectoryBalance

# 🌟 核心导入：集结三大预言机 (AMP, Hemo, HMoment)
from gflownet.tasks.oracle import (
    AMPOracle,
    BERTMICOracle,
    HemoOracle,
    HydrophobicMomentOracle,
    MBCMICOracle,
    ToxOracle,
    preflight_reward_oracles,
)
from gflownet.utils.multiobjective_hooks import MultiObjectiveStatsHook
from gflownet.utils.reproducibility import build_amp_reproducibility_manifest


class AMPSequenceDataset(Dataset):
    def __init__(self, seqs, flat_rewards):
        self.seqs = seqs
        self.flat_rewards = flat_rewards.float()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.flat_rewards[idx]


def read_fasta_sequences(path):
    seqs = []
    cur = ""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append(cur.upper())
                    cur = ""
            else:
                cur += line

    if cur:
        seqs.append(cur.upper())

    return seqs


# =====================================================================
# 1. 任务类 (完美适配单层序列的动态多目标架构)
# =====================================================================
class AMPMOOTask(SEHMOOTask):
    def __init__(self, cfg, rng, wrap_model=None,mic_oracle_name="mbc"):
        self.mic_oracle_name = mic_oracle_name
        self.min_length = 5
        # 1. 动态获取你传入的真实目标列表 (2个或3个)
        self.real_objs = list(cfg.task.seh_moo.objectives)
        # 2. 动态截取等量的占位符骗过底层
        cfg.task.seh_moo.objectives = ["seh", "qed", "sa", "mw"][:len(self.real_objs)]
        
        super().__init__(dataset=None, cfg=cfg, rng=rng, wrap_model=wrap_model)
        
        # ==========================================================
        # 🌟 自适应维度探测与伸缩
        # ==========================================================
        if hasattr(self, 'pref_cond') and self.pref_cond is not None:
            self.pref_cond.num_objectives = len(self.real_objs)
        if hasattr(self, 'focus_cond') and self.focus_cond is not None:
            self.focus_cond.num_objectives = len(self.real_objs)

        try:
            dummy_cond = self.sample_conditional_information(1)
            self.num_cond_dim = dummy_cond["encoding"].shape[1]
        except Exception:
            self.num_cond_dim = 64 + (len(self.real_objs) - 2) * 16 
        
        # 恢复真实菜单名
        self.objectives = self.real_objs
        cfg.task.seh_moo.objectives = self.real_objs

        # 只初始化本次训练真正选择的奖励预测器。
        self.amp_oracle = AMPOracle(device="cpu") if "amp" in self.objectives else None
        self.hemo_oracle = HemoOracle(device="cpu") if "hemo" in self.objectives else None
        self.hmoment_oracle = (
            HydrophobicMomentOracle(device="cpu")
            if "hmoment" in self.objectives
            else None
        )
        if "mic" in self.objectives:
            mic_oracle_name = getattr(self, "mic_oracle_name", "mbc")

            if mic_oracle_name == "mbc":
                self.mic_oracle = MBCMICOracle(output_key="pmic")
                print("[MIC Oracle] using MBCMICOracle(output_key='pmic')")
            elif mic_oracle_name == "bert":
                self.mic_oracle = BERTMICOracle(output_key="mic_score")
                print("[MIC Oracle] using BERTMICOracle(output_key='mic_score')")
            else:
                raise ValueError(f"Unknown mic_oracle: {mic_oracle_name}")
        else:
            self.mic_oracle = None

        if "tox" in self.objectives:
            self.tox_oracle = ToxOracle(device="cpu")
        else:
            self.tox_oracle = None

        self.vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    def _load_task_models(self):
        # 彻底拦截父类网络请求，不下载化学模型
        return {}

    # 🎯 动态打分工厂 (专为一维 List[int] 状态定制)
    def compute_flat_rewards(self, seqs: List[List[int]]) -> Tuple[FlatRewards, Tensor]:
        # 1. 数字序列解码为氨基酸字符串
        str_seqs = []
        for s in seqs:
            # 过滤掉可能的 Stop 动作 (20) 或 Pad (21)
            clean_s = [idx for idx in s if idx < 20] 
            str_seqs.append("".join([self.vocab[idx] for idx in clean_s]))

        flat_r: List[Tensor] = []
        
        # 保持与高级版完全一致的惩罚力度！
        # def get_penalty(seq_str, target_len=15, sigma=25):
        #     L = len(seq_str)
        #     penalty = math.exp(- ((L - target_len) ** 2) / (2 * sigma ** 2))
        #     return max(0.1, penalty)
                
        # penalties = torch.tensor([get_penalty(s) for s in str_seqs], dtype=torch.float32)

        # # 关闭长度软约束：所有序列的长度惩罚都设为 1
        penalties = torch.ones(len(str_seqs), dtype=torch.float32)

        with torch.no_grad():
            # additive_penalties = 4.34 * torch.log(1.0 / penalties)

            # 🌟 核心开关机制
            obj_scores = {}
            
            if "amp" in self.objectives:
                amp_scores = self.amp_oracle.predict(str_seqs).cpu()
                obj_scores["amp"] = amp_scores * penalties

            if "hemo" in self.objectives:
                hemo_probs = self.hemo_oracle.predict(str_seqs).cpu()
                hemo_scores = 1.0 - hemo_probs  
                obj_scores["hemo"] = hemo_scores * penalties

            if "hmoment" in self.objectives:
                hm_scores = self.hmoment_oracle.predict(str_seqs).cpu()
                obj_scores["hmoment"] = torch.clamp(hm_scores / 0.6, max=1.0) * penalties

            if "mic" in self.objectives:
                if self.mic_oracle is None:
                    raise RuntimeError("mic objective is enabled, but mic_oracle is not initialized.")

                mic_scores = self.mic_oracle.predict(str_seqs).cpu()
                obj_scores["mic"] = mic_scores * penalties

            if "tox" in self.objectives:
                if self.tox_oracle is None:
                    raise RuntimeError("tox objective is enabled, but tox_oracle is not initialized.")

                tox_probs = self.tox_oracle.predict(str_seqs).cpu()

                # ToxinPred3 的分数越高越毒；OP-GFN 需要越大越好，所以这里转成安全分数。
                tox_scores = 1.0 - tox_probs
                tox_scores = torch.clamp(tox_scores, min=0.0, max=1.0)

                obj_scores["tox"] = tox_scores * penalties

            # 🌟 动态按顺序出餐
            for obj_name in self.objectives:
                if obj_name in obj_scores:
                    flat_r.append(obj_scores[obj_name])
                else:
                    raise ValueError(f"预言机中未定义目标: {obj_name}，请检查启动命令！")

            if len(flat_r) > 0:
                flat_rewards = torch.stack(flat_r, dim=1).detach().cpu()
            else:
                flat_rewards = torch.zeros((len(seqs), len(self.objectives)))

        # 合法性校验：去除Stop符后长度必须大于等于 5
        is_valid = torch.tensor(
            [len([idx for idx in s if idx < 20]) >= self.min_length for s in seqs],
            dtype=torch.bool,
        )
        
        return FlatRewards(flat_rewards.float()), is_valid


# =====================================================================
# 2. 训练器类
# =====================================================================
class AMPMOOTrainer(StandardOnlineTrainer):
    def __init__(self, hps, args):
        self.cmd_args = args

        seed = getattr(args, "seed", 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        super().__init__(hps) 
    def setup_data(self):
        """
        先初始化为空数据集。

        注意：
        offline_data 不能在这里加载，因为这里执行时 self.ctx 还没有创建。
        真正的 offline_data 加载放到 setup_env_context() 后面。
        """
        self.training_data = []
        self.test_data = []
        self.offline_data_path = None
        self.reward_cache_path = None
    def setup_offline_data(self):
        offline_path = getattr(self.cmd_args, "offline_data", None)
        if not offline_path:
            return

        offline_path = Path(offline_path)
        if not offline_path.exists():
            raise FileNotFoundError(f"Offline data not found: {offline_path}")
        self.offline_data_path = offline_path

        raw_seqs = read_fasta_sequences(offline_path)

        seqs = []
        token_seqs = []
        for seq in raw_seqs:
            token_seq = self.ctx.seq_str_to_idx(seq)

            # 确保没有非法氨基酸，并且长度不超过生成上限
            if (
                len(token_seq) == len(seq)
                and self.ctx.min_length <= len(token_seq) <= self.cfg.algo.max_nodes
            ):
                seqs.append(seq)
                token_seqs.append(token_seq)

        if not token_seqs:
            raise ValueError(f"No valid offline AMP sequences found in {offline_path}")

        obj_key = "_".join(self.task.objectives)
        cache_tag = getattr(self.cmd_args, "reward_cache_tag", "")
        if cache_tag:
            cache_path = offline_path.with_name(
                f"{offline_path.stem}_{obj_key}_{cache_tag}_rewards.pt"
            )
        else:
            cache_path = offline_path.with_name(
                f"{offline_path.stem}_{obj_key}_rewards.pt"
            )
        self.reward_cache_path = cache_path

        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu")
            seqs = cache["seqs"]
            flat_rewards = cache["flat_rewards"]
            print(f"[Offline AMP] loaded cached rewards from {cache_path}")
        else:
            chunks = []
            bs = getattr(self.cmd_args, "offline_reward_batch_size", 64)

            for i in range(0, len(token_seqs), bs):
                flat_r, is_valid = self.task.compute_flat_rewards(token_seqs[i:i + bs])
                chunks.append(torch.as_tensor(flat_r).cpu())
                print(f"[Offline AMP] reward batch {i + len(token_seqs[i:i + bs])}/{len(token_seqs)}")

            flat_rewards = torch.cat(chunks, dim=0)
            torch.save(
                {
                    "seqs": seqs,
                    "flat_rewards": flat_rewards,
                    "objectives": list(self.task.objectives),
                },
                cache_path,
            )
            print(f"[Offline AMP] cached rewards to {cache_path}")

        self.training_data = AMPSequenceDataset(seqs, flat_rewards)
        print(f"[Offline AMP] loaded {len(self.training_data)} offline short peptides")

    def build_reproducibility_manifest(self):
        """Return the exact non-weight state needed to reproduce this run."""
        return build_amp_reproducibility_manifest(
            cfg=self.cfg,
            cmd_args=self.cmd_args,
            task=self.task,
            ctx=self.ctx,
            model=self.model,
            algo=self.algo,
            offline_data_path=self.offline_data_path,
            reward_cache_path=self.reward_cache_path,
        )

    def set_default_hps(self, cfg):
        pass
        
    def setup(self):
        super().setup()
        moo_hook = MultiObjectiveStatsHook
        # 多目标顶会评估雷达
        moo_hook = MultiObjectiveStatsHook(
            num_to_keep=2000,
            log_dir=self.cfg.log_dir,
            save_every=50,
            compute_hvi=self.cmd_args.compute_hvi,            
            compute_igd=self.cmd_args.compute_igd,
            compute_pc_entropy=self.cmd_args.compute_pc_entropy
        )
        
        if hasattr(self, 'sampling_hooks'):
            self.sampling_hooks.append(moo_hook)
        else:
            self.sampling_hooks = [moo_hook]
        print("[Hook] multi-objective statistics hook enabled")

    def setup_task(self):
        self.task = AMPMOOTask(
            cfg=self.cfg,
            rng=self.rng,
            mic_oracle_name=getattr(self.cmd_args, "mic_oracle", "mbc"),
        )
        self.num_cond_dim = self.task.num_cond_dim

    def setup_env_context(self):
        """
        创建 AMP 序列环境。

        新增：
        1. motif_vocab_path：读取 Top-100 motif
        2. max_motif_actions：最多允许使用几个 motif，默认 2
        3. min_length：最短生成长度，默认 5
        """
        self.ctx = SeqBuildingEnvContext(
            max_length=self.cfg.algo.max_nodes,
            num_cond_dim=self.num_cond_dim,
            motif_vocab_path=getattr(self.cmd_args, "motif_vocab_path", None),
            max_motif_actions=getattr(self.cmd_args, "max_motif_actions", 2),
            min_length=getattr(self.cmd_args, "min_length", 5),
        )
        self.task.min_length = self.ctx.min_length

        # ctx 已经创建，此时可以安全加载 offline_data
        self.setup_offline_data()

    def setup_model(self):
        self.model = SeqTransformerGFN(
            vocab_size=20, 
            max_len=self.cfg.algo.max_nodes,
            d_model=self.cfg.model.num_emb,
            nhead=self.cfg.model.graph_transformer.num_heads,
            num_layers=self.cfg.model.num_layers,
            num_cond_dim=self.num_cond_dim,
            # 新增：让模型输出维度和环境动作空间一致
            num_actions=self.ctx.num_actions,
            pad_idx=self.ctx.pad_idx,
            max_motif_actions=self.ctx.max_motif_actions,
            # 新增：双门控开关和强度
            use_motif_gate=getattr(self.cmd_args, "use_motif_gate", False),
            motif_use_gate_strength=getattr(self.cmd_args, "motif_use_gate_strength", 1.0),
            motif_select_gate_strength=getattr(self.cmd_args, "motif_select_gate_strength", 1.0),
        ).to(self.device if hasattr(self, 'device') else "cuda")
        
        if self.num_cond_dim > 0:
            self.model.logZ = nn.Sequential(
                nn.Linear(self.num_cond_dim, self.cfg.model.num_emb * 2),
                nn.ReLU(),
                nn.Linear(self.cfg.model.num_emb * 2, 1)
            ).to(self.device if hasattr(self, 'device') else "cpu")
        else:
            self.model.logZ_param = nn.Parameter(torch.zeros(1))
            self.model.logZ = lambda cond_info: self.model.logZ_param.expand(cond_info.shape[0], 1)

    def setup_algo(self):
        self.algo = SeqTrajectoryBalance(
            self.ctx, 
            self.cfg, 
            max_nodes=self.cfg.algo.max_nodes
        )
        if self.cfg.algo.method == "TB":
            self.algo.set_is_conditioned(self.num_cond_dim > 0)
        # 将命令行里的开关传递给底层算法
        self.algo.use_global_rank = getattr(self.cmd_args, 'use_global_rank', False)

        self.algo.motif_usage_loss_weight = getattr(
            self.cmd_args,
            "motif_usage_loss_weight",
            0.0
        )

        self.algo.pareto_len_limit = getattr(
            self.cmd_args,
            "pareto_len_limit",
            30,
        )

        self.algo.replay_len_limit = getattr(
            self.cmd_args,
            "replay_len_limit",
            30,
        )

        print(
            f"[Motif Control] gate={getattr(self.cmd_args, 'use_motif_gate', False)}, "
            f"use_gate_strength={getattr(self.cmd_args, 'motif_use_gate_strength', 1.0)}, "
            f"select_gate_strength={getattr(self.cmd_args, 'motif_select_gate_strength', 1.0)}, "
            f"usage_loss_weight={self.algo.motif_usage_loss_weight}, "
            f"pareto_len_limit={self.algo.pareto_len_limit}, "
            f"replay_len_limit={self.algo.replay_len_limit}"
        )

# =====================================================================
# 3. 运行主干配置
# =====================================================================
def main(args):
    if isinstance(args.objectives, str):
        args.objectives = args.objectives.split()

    if not args.skip_oracle_preflight:
        scores = preflight_reward_oracles(
            objectives=args.objectives,
            mic_oracle_name=args.mic_oracle,
        )
        print(f"[Oracle preflight] all selected reward services passed: {scores}")

    hps = {
        "log_dir": args.log_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "seed": args.seed,
        "num_training_steps": args.num_training_steps, # 配合你长周期的跑图计划
        "num_final_gen_steps": 50,
        "validate_every": 200,
        "num_workers": 0,
        "algo": {
            "global_batch_size": args.global_batch_size,
            "offline_ratio":args.offline_ratio if args.offline_data else 0.0,        
            "valid_offline_ratio": 0.0,   
            "method": "TB",
            "max_nodes": 50, 
            "sampling_tau": 0.95,
            "train_random_action_prob": args.train_random_action_prob,
            "tb": {
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50000,
                "do_ordering": True if args.type == 'ordering' else False,
            },
        },
        "model": {
            "num_layers": 4,
            "num_emb": 256,
            "graph_transformer": {"num_heads": 8}, 
        },
        "task": {
            "seh_moo": { 
                "n_valid": 15, 
                "n_valid_repeats": 128, 
                "objectives": args.objectives, 
                "preference_type": "dirichlet" if args.type == 'pref' else None, 
                "focus_type": None
            }
        },
        "cond": {
            "temperature": {"sample_dist": "constant", "dist_params": [60.0], "num_thermometer_dim": 32},
            "focus_region": {"focus_type": None, "use_steer_thermomether": True},
            "weighted_prefs": {"preference_type": "dirichlet" if args.type == 'pref' else None},
        },
        "replay": {"use": args.replay, "capacity": 100000, "warmup": 1000,"hindsight_ratio": 0.0,},
    }
    
    if os.path.exists(hps["log_dir"]) and hps["overwrite_existing_exp"]:
        shutil.rmtree(hps["log_dir"])
    os.makedirs(hps["log_dir"], exist_ok=True)

    trial = AMPMOOTrainer(hps, args)
    trial.print_every = 10
    trial.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        default="./outputs/training/paper_mic_hemo_10000_off020_rand015_seed0",
        type=str,
    )
    
    # 默认使用与论文方法一致的 BERT-MIC + Hemo 双目标配置。
    parser.add_argument(
        "--objectives",
        default=["mic", "hemo"],
        nargs="+",
        type=str,
        help="优化目标，可选: amp hemo hmoment mic tox",
    )
    
    parser.add_argument(
        "--replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--type", default='ordering', choices=['pref', 'goal', 'ordering'])
    
    parser.add_argument("--use_global_rank", action="store_true", help="启用多级软标签 (ICLR 2025 路线 B)")
    parser.add_argument("--compute_hvi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute_igd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compute_pc_entropy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    
    parser.add_argument(
        "--offline_data",
        default="./datasets/offline/bert_ec_final_offline_mic_priority_top2000.fasta",
        type=str,
    )
    parser.add_argument("--offline_ratio", default=0.20, type=float)
    parser.add_argument("--offline_reward_batch_size", default=64, type=int)
    parser.add_argument(
        "--global_batch_size",
        default=64,
        type=int,
        help=(
            "训练轨迹 batch size。0627 历史训练为 64；"
            "它与 offline_reward_batch_size 是两个不同参数。"
        ),
    )
    parser.add_argument(
        "--mic_oracle",
        default="bert",
        choices=["mbc", "bert"],
        type=str,
        help="MIC oracle backend: mbc uses MBC_MIC_PORT/5011, bert uses MIC_BERT_PORT/5010.",
    )


    # ===============================
    # 单层 motif 动作空间参数
    # ===============================
    parser.add_argument(
        "--motif_vocab_path",
        default="./datasets/motifs/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt",
        type=str,
        help="Top motif 词表路径，每行一个 motif。",
    )

    parser.add_argument("--max_motif_actions",default=2,type=int,help="每条生成轨迹最多允许使用几个 motif 动作。默认 2。")


    parser.add_argument("--min_length",default=5,type=int,help="最短肽链长度。短于该长度不允许 STOP。默认 5。")

    parser.add_argument("--train_random_action_prob",default=0.15,type=float,help="训练时随机探索概率。")

    parser.add_argument("--num_training_steps",default=10000,type=int,help="训练步数。debug 时可以设为 50 或 100。")
    parser.add_argument(
        "--use_motif_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 motif 双门控。不开这个参数时，模型保持原来的 motif 动作打分方式。"
    )

    parser.add_argument(
        "--motif_use_gate_strength",
        default=1.0,
        type=float,
        help="motif 总开关门控强度。越大，模型越能整体提高或压低 motif 动作。"
    )

    parser.add_argument(
        "--motif_select_gate_strength",
        default=1.0,
        type=float,
        help="单个 motif 选择门控强度。越大，模型越能区分不同 motif。"
    )

    parser.add_argument(
        "--motif_usage_loss_weight",
        default=0.0,
        type=float,
        help="motif 使用惩罚权重。越大，越抑制模型过度使用 motif。0 表示关闭。"
    )

    parser.add_argument(
        "--pareto_len_limit",
        default=30,
        type=int,
        help="长度超过该值的序列不允许作为训练 batch 的 Pareto 正样本，但不修改 MIC/HEMO reward。",
    )

    parser.add_argument(
        "--reward_cache_tag",
        default="bert_ec_final_mic_priority_motif67",
        type=str,
        help="离线 reward 缓存标签，用于区分不同 MIC oracle，例如 bertmic、esm2mic。",
    )
    parser.add_argument(
        "--replay_len_limit",
        default=30,
        type=int,
        help="长度超过该值的 online 序列不放入 replay buffer。",
    )
    parser.add_argument(
        "--skip_oracle_preflight",
        action="store_true",
        help=(
            "跳过训练前的真实奖励预测检查。仅建议纯离线调试使用；"
            "在线训练应保持预检开启。"
        ),
    )
    args = parser.parse_args()
    main(args)
