#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从一个已经训练完成的 MiOP-GFN checkpoint 中反复采样。

要点：
1. 优先加载 EMA sampling_model 权重。
2. 同一个 checkpoint 可用任意 sampling seed 重复采样。
3. 只做生成环境本身的合法动作约束，不做 MIC/HEMO/复杂度过滤。
4. 输出未经 oracle 重打分或后处理的原始候选池 CSV。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gflownet.algo.seq_trajectory_balance import SeqTrajectoryBalance
from gflownet.envs.seq_env import SeqBuildingEnvContext
from gflownet.models.seq_model import SeqTransformerGFN
from gflownet.tasks.amp_moo import AMPMOOTask
from gflownet.utils.reproducibility import (
    ReproducibilityManifestError,
    validate_amp_sampling_manifest,
)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """兼容新旧 PyTorch 的 checkpoint 加载方式。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_logz(model: nn.Module, num_cond_dim: int, num_emb: int) -> None:
    if num_cond_dim <= 0:
        raise ValueError(
            "当前独立采样脚本面向有条件的 MOO 模型，要求 num_cond_dim > 0。"
        )
    model.logZ = nn.Sequential(
        nn.Linear(num_cond_dim, num_emb * 2),
        nn.ReLU(),
        nn.Linear(num_emb * 2, 1),
    )


def choose_state_dict(
    checkpoint: Dict[str, Any],
    weights: str,
) -> tuple[Dict[str, torch.Tensor], str]:
    has_ema = "sampling_model_state_dict" in checkpoint

    if weights == "ema":
        if not has_ema:
            raise KeyError(
                "checkpoint 中没有 sampling_model_state_dict。"
                "旧 checkpoint 只能使用 --weights model；"
                "若要精确恢复训练时的 EMA 采样模型，需要重新训练或在训练进程退出前保存 EMA。"
            )
        return checkpoint["sampling_model_state_dict"], "EMA sampling_model"

    if weights == "model":
        return checkpoint["models_state_dict"][0], "ordinary training model"

    if has_ema:
        return checkpoint["sampling_model_state_dict"], "EMA sampling_model"

    return checkpoint["models_state_dict"][0], "ordinary training model (EMA unavailable)"


def main(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num_samples 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch_size 必须大于 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，自动切换到 CPU。")
        device = torch.device("cpu")

    seed_everything(args.sampling_seed)

    checkpoint = load_checkpoint(args.checkpoint)
    if "cfg" not in checkpoint:
        raise KeyError("checkpoint 缺少 cfg，无法重建模型。")
    cfg = checkpoint["cfg"]

    manifest = checkpoint.get("reproducibility_manifest")
    if manifest is None:
        if not args.allow_legacy_checkpoint:
            raise ReproducibilityManifestError(
                "checkpoint 缺少 reproducibility_manifest，拒绝在无法校验 motif 语义的"
                "情况下采样。如果你确认这是旧 checkpoint 且愿意承担风险，显式加上 "
                "--allow_legacy_checkpoint。"
            )
        print(
            "[Warning] legacy checkpoint: reproducibility_manifest unavailable; "
            "only tensor/action-count checks will be enforced."
        )

    # AMPMOOTask 初始化时会探测一次 conditional encoding；初始化完成后重置 RNG，
    # 保证正式采样流严格从 sampling_seed 开始。
    task = AMPMOOTask(
        cfg=cfg,
        rng=np.random.default_rng(args.sampling_seed),
        mic_oracle_name=args.mic_oracle,
    )
    task.rng = np.random.default_rng(args.sampling_seed)
    num_cond_dim = int(task.num_cond_dim)

    ctx = SeqBuildingEnvContext(
        max_length=int(cfg.algo.max_nodes),
        num_cond_dim=num_cond_dim,
        motif_vocab_path=args.motif_vocab_path,
        max_motif_actions=args.max_motif_actions,
        min_length=args.min_length,
    )

    model = SeqTransformerGFN(
        vocab_size=20,
        max_len=int(cfg.algo.max_nodes),
        d_model=int(cfg.model.num_emb),
        nhead=int(cfg.model.graph_transformer.num_heads),
        num_layers=int(cfg.model.num_layers),
        num_cond_dim=num_cond_dim,
        num_actions=ctx.num_actions,
        pad_idx=ctx.pad_idx,
        max_motif_actions=ctx.max_motif_actions,
        use_motif_gate=args.use_motif_gate,
        motif_use_gate_strength=args.motif_use_gate_strength,
        motif_select_gate_strength=args.motif_select_gate_strength,
    )
    build_logz(model, num_cond_dim, int(cfg.model.num_emb))

    manifest_hash = None
    if manifest is not None:
        manifest_hash = validate_amp_sampling_manifest(
            manifest,
            motif_vocab_path=args.motif_vocab_path,
            task=task,
            ctx=ctx,
            model=model,
        )
        print(f"[Manifest] strict validation passed: sha256={manifest_hash}")

    state_dict, state_name = choose_state_dict(checkpoint, args.weights)

    # 提前检查动作空间是否与 checkpoint 一致，避免 motif 词表不一致时给出难读的报错。
    action_weight_key = "action_head.weight"
    if action_weight_key in state_dict:
        expected_actions = int(state_dict[action_weight_key].shape[0])
        if expected_actions != ctx.num_actions:
            raise ValueError(
                "动作空间与 checkpoint 不一致："
                f"checkpoint num_actions={expected_actions}，"
                f"当前 motif 文件构建出的 num_actions={ctx.num_actions}。"
                "请使用训练时完全相同的 motif_vocab_path 和 max_motif_actions。"
            )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    algo = SeqTrajectoryBalance(
        ctx,
        cfg,
        max_nodes=int(cfg.algo.max_nodes),
    )

    # 独立 PyTorch RNG：真正控制 torch.multinomial 的动作抽样。
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    torch_generator = torch.Generator(device=generator_device)
    torch_generator.manual_seed(args.sampling_seed)

    checkpoint_step = int(checkpoint.get("step", 0))
    rows = []
    generated = 0

    print(f"[Checkpoint] loaded: {state_name}")
    print(f"[Sampling] seed={args.sampling_seed}, target={args.num_samples}, device={device}")

    with torch.no_grad():
        while generated < args.num_samples:
            current_n = min(args.batch_size, args.num_samples - generated)

            cond_info = task.sample_conditional_information(
                current_n,
                checkpoint_step,
            )

            trajs = algo.create_training_data_from_own_samples(
                model=model,
                n=current_n,
                cond_info=cond_info["encoding"],
                random_action_prob=0.0,
                torch_generator=torch_generator,
            )

            for traj in trajs:
                sequence = ctx.idx_to_seq_str(traj["result"])
                rows.append(
                    {
                        "sequence": sequence,
                        "seq_len": len(sequence),
                        "sampling_seed": args.sampling_seed,
                        "checkpoint_step": checkpoint_step,
                        "weight_source": state_name,
                        "reproducibility_manifest_sha256": manifest_hash or "legacy-unverified",
                        "is_valid": bool(traj.get("is_valid", True)),
                        "motif_count": int(traj.get("motif_count", 0)),
                        "num_aa_actions": int(traj.get("num_aa_actions", 0)),
                        "num_motif_actions": int(traj.get("num_motif_actions", 0)),
                        "num_total_actions": int(traj.get("num_total_actions", 0)),
                        "motif_action_ids": ",".join(
                            map(str, traj.get("motif_action_ids", []))
                        ),
                        "motif_action_names": ";".join(
                            traj.get("motif_action_names", [])
                        ),
                    }
                )

            generated += current_n
            print(f"[Sampling] {generated}/{args.num_samples}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"[Done] output={output_path}")
    print(f"[Done] rows={len(df)}")
    print(f"[Done] unique_sequences={df['sequence'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repeated sampling from a trained MiOP-GFN checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--motif_vocab_path", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    parser.add_argument("--sampling_seed", required=True, type=int)
    parser.add_argument("--num_samples", default=5000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument(
        "--weights",
        default="auto",
        choices=["auto", "ema", "model"],
        help="auto 优先 EMA；旧 checkpoint 无 EMA 时回退到普通训练模型。",
    )
    parser.add_argument(
        "--mic_oracle",
        default="bert",
        choices=["bert", "mbc"],
        help="只用于重建 AMPMOOTask；纯采样阶段不会调用 oracle 打分。",
    )
    parser.add_argument("--max_motif_actions", default=2, type=int)
    parser.add_argument("--min_length", default=5, type=int)
    parser.add_argument("--use_motif_gate", action="store_true")
    parser.add_argument("--motif_use_gate_strength", default=1.0, type=float)
    parser.add_argument("--motif_select_gate_strength", default=1.0, type=float)
    parser.add_argument(
        "--allow_legacy_checkpoint",
        action="store_true",
        help=(
            "允许缺少 reproducibility_manifest 的旧 checkpoint；仅保留张量形状和动作数检查，"
            "无法证明 motif 顺序、gate 强度和长度阈值与训练一致。"
        ),
    )

    main(parser.parse_args())
