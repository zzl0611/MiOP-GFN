"""Reproducibility metadata for AMP-MOO training and sampling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


AMP_MANIFEST_SCHEMA_VERSION = 1


class ReproducibilityManifestError(ValueError):
    """Raised when a checkpoint cannot reproduce the requested sampler."""


def sha256_file(path: Union[str, Path]) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data using a stable serialization."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Describe an input artifact while retaining its configured path."""
    if path is None or str(path) == "":
        return {
            "configured_path": None,
            "resolved_path": None,
            "sha256": None,
            "size_bytes": None,
        }

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Cannot fingerprint missing file: {artifact}")
    return {
        "configured_path": str(path),
        "resolved_path": str(artifact.resolve()),
        "sha256": sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def build_sequence_action_mapping(ctx: Any) -> list[Dict[str, Any]]:
    """Build the exact semantic mapping represented by each action logit."""
    mapping = [
        {"action_id": idx, "kind": "amino_acid", "token": aa}
        for idx, aa in enumerate(ctx.vocab)
    ]
    mapping.extend(
        {
            "action_id": ctx.motif_action_start + idx,
            "kind": "motif",
            "token": motif,
        }
        for idx, motif in enumerate(ctx.motifs)
    )
    mapping.append(
        {
            "action_id": ctx.stop_action_idx,
            "kind": "stop",
            "token": "<STOP>",
        }
    )
    return mapping


def build_amp_reproducibility_manifest(
    *,
    cfg: Any,
    cmd_args: Any,
    task: Any,
    ctx: Any,
    model: Any,
    algo: Any,
    offline_data_path: Optional[Union[str, Path]],
    reward_cache_path: Optional[Union[str, Path]],
) -> Dict[str, Any]:
    """Capture all non-weight state required to reconstruct AMP-MOO training."""
    motif_path = getattr(cmd_args, "motif_vocab_path", None)
    effective_motifs = list(ctx.motifs)
    action_mapping = build_sequence_action_mapping(ctx)

    manifest: Dict[str, Any] = {
        "schema_version": AMP_MANIFEST_SCHEMA_VERSION,
        "manifest_type": "amp_moo_training",
        "motif_vocab": {
            **file_fingerprint(motif_path),
            "effective_motifs": effective_motifs,
            "effective_motifs_sha256": canonical_sha256(effective_motifs),
        },
        "offline_data": file_fingerprint(offline_data_path),
        "reward_cache": {
            **file_fingerprint(reward_cache_path),
            "tag": str(getattr(cmd_args, "reward_cache_tag", "")),
        },
        "task": {
            "objectives": list(task.objectives),
            "mic_oracle": str(getattr(task, "mic_oracle_name", "")),
            "num_cond_dim": int(getattr(task, "num_cond_dim")),
        },
        "environment": {
            "amino_acid_vocab": list(ctx.vocab),
            "max_length": int(ctx.max_length),
            "min_length": int(ctx.min_length),
            "max_motif_actions": int(ctx.max_motif_actions),
            "pad_idx": int(ctx.pad_idx),
            "stop_action_idx": int(ctx.stop_action_idx),
        },
        "action_space": {
            "num_actions": int(ctx.num_actions),
            "mapping": action_mapping,
            "mapping_sha256": canonical_sha256(action_mapping),
        },
        "model": {
            "num_emb": int(cfg.model.num_emb),
            "num_layers": int(cfg.model.num_layers),
            "num_heads": int(cfg.model.graph_transformer.num_heads),
            "use_motif_gate": bool(model.use_motif_gate),
            "motif_use_gate_strength": float(model.motif_use_gate_strength),
            "motif_select_gate_strength": float(model.motif_select_gate_strength),
        },
        "losses": {
            "motif_usage_loss_weight": float(algo.motif_usage_loss_weight),
            "pareto_len_limit": int(algo.pareto_len_limit),
        },
        "replay": {
            "enabled": bool(cfg.replay.use),
            "capacity": int(cfg.replay.capacity),
            "warmup": int(cfg.replay.warmup),
            "hindsight_ratio": float(cfg.replay.hindsight_ratio),
            "length_limit": int(algo.replay_len_limit),
        },
        "sampling_training": {
            "offline_ratio": float(cfg.algo.offline_ratio),
            "train_random_action_prob": float(cfg.algo.train_random_action_prob),
            "global_batch_size": int(cfg.algo.global_batch_size),
            "sampling_tau": float(cfg.algo.sampling_tau),
            "seed": int(cfg.seed),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def validate_manifest_integrity(manifest: Mapping[str, Any]) -> str:
    """Validate schema and the manifest's own content hash."""
    if manifest.get("schema_version") != AMP_MANIFEST_SCHEMA_VERSION:
        raise ReproducibilityManifestError(
            "Unsupported reproducibility_manifest schema_version: "
            f"{manifest.get('schema_version')!r}; expected {AMP_MANIFEST_SCHEMA_VERSION}."
        )
    if manifest.get("manifest_type") != "amp_moo_training":
        raise ReproducibilityManifestError(
            "Checkpoint reproducibility_manifest is not an AMP-MOO manifest."
        )

    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        raise ReproducibilityManifestError(
            "reproducibility_manifest is missing manifest_sha256."
        )
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    observed = canonical_sha256(unhashed)
    if observed != expected:
        raise ReproducibilityManifestError(
            "reproducibility_manifest content hash mismatch: "
            f"checkpoint={expected}, recomputed={observed}."
        )
    return expected


def _require_equal(label: str, expected: Any, observed: Any) -> None:
    if observed != expected:
        raise ReproducibilityManifestError(
            f"{label} mismatch: checkpoint={expected!r}, sampler={observed!r}."
        )


def _require_float_equal(label: str, expected: Any, observed: Any) -> None:
    try:
        equal = math.isclose(
            float(expected),
            float(observed),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        equal = False
    if not equal:
        raise ReproducibilityManifestError(
            f"{label} mismatch: checkpoint={expected!r}, sampler={observed!r}."
        )


def validate_amp_sampling_manifest(
    manifest: Mapping[str, Any],
    *,
    motif_vocab_path: Union[str, Path],
    task: Any,
    ctx: Any,
    model: Any,
) -> str:
    """Fail closed unless sampler semantics exactly match the training manifest."""
    manifest_hash = validate_manifest_integrity(manifest)

    motif = manifest.get("motif_vocab", {})
    expected_file_hash = motif.get("sha256")
    if not expected_file_hash:
        raise ReproducibilityManifestError(
            "Training manifest does not contain a motif vocabulary SHA-256."
        )
    observed_file_hash = sha256_file(motif_vocab_path)
    _require_equal("motif_vocab.sha256", expected_file_hash, observed_file_hash)

    observed_motifs = list(ctx.motifs)
    _require_equal(
        "motif_vocab.effective_motifs",
        motif.get("effective_motifs"),
        observed_motifs,
    )
    _require_equal(
        "motif_vocab.effective_motifs_sha256",
        motif.get("effective_motifs_sha256"),
        canonical_sha256(observed_motifs),
    )

    task_manifest = manifest.get("task", {})
    _require_equal("task.objectives", task_manifest.get("objectives"), list(task.objectives))
    _require_equal("task.mic_oracle", task_manifest.get("mic_oracle"), task.mic_oracle_name)
    _require_equal("task.num_cond_dim", task_manifest.get("num_cond_dim"), task.num_cond_dim)

    environment = manifest.get("environment", {})
    environment_checks = {
        "amino_acid_vocab": list(ctx.vocab),
        "max_length": int(ctx.max_length),
        "min_length": int(ctx.min_length),
        "max_motif_actions": int(ctx.max_motif_actions),
        "pad_idx": int(ctx.pad_idx),
        "stop_action_idx": int(ctx.stop_action_idx),
    }
    for key, observed in environment_checks.items():
        _require_equal(f"environment.{key}", environment.get(key), observed)

    action_space = manifest.get("action_space", {})
    observed_mapping = build_sequence_action_mapping(ctx)
    _require_equal("action_space.num_actions", action_space.get("num_actions"), ctx.num_actions)
    _require_equal("action_space.mapping", action_space.get("mapping"), observed_mapping)
    _require_equal(
        "action_space.mapping_sha256",
        action_space.get("mapping_sha256"),
        canonical_sha256(observed_mapping),
    )

    model_manifest = manifest.get("model", {})
    architecture_checks = {
        "num_emb": int(model.embedding.embedding_dim),
        "num_layers": len(model.transformer.layers),
        "num_heads": int(model.transformer.layers[0].self_attn.num_heads),
    }
    for key, observed in architecture_checks.items():
        _require_equal(f"model.{key}", model_manifest.get(key), observed)
    _require_equal(
        "model.use_motif_gate",
        model_manifest.get("use_motif_gate"),
        bool(model.use_motif_gate),
    )
    _require_float_equal(
        "model.motif_use_gate_strength",
        model_manifest.get("motif_use_gate_strength"),
        model.motif_use_gate_strength,
    )
    _require_float_equal(
        "model.motif_select_gate_strength",
        model_manifest.get("motif_select_gate_strength"),
        model.motif_select_gate_strength,
    )

    return manifest_hash
