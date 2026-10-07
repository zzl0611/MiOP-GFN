#!/usr/bin/env python3
"""Unified command-line entry point for the AMP MiOP-GFN project."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
DATASET_ROOT = REPO_ROOT / "datasets"
PREDICTOR_ROOT = REPO_ROOT / "predictors"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
TEST_ROOT = REPO_ROOT / "tests"
CONFIG_ROOT = REPO_ROOT / "configs" / "train"
MIC_SERVER = PREDICTOR_ROOT / "bert_mic" / "server.py"

TRAIN_IMPORTS = {
    "torch": "pytorch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "requests": "requests",
    "flask": "flask",
    "transformers": "transformers",
    "rdkit": "rdkit",
    "omegaconf": "omegaconf",
    "torch_geometric": "torch-geometric",
    "botorch": "botorch",
    "gpytorch": "gpytorch",
    "modlamp": "modlamp",
}


def load_preset(name: str) -> dict:
    path = CONFIG_ROOT / f"{name}.json"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in CONFIG_ROOT.glob("*.json")))
        raise SystemExit(f"Unknown preset {name!r}. Available presets: {available}")
    with path.open("r", encoding="utf-8") as handle:
        preset = json.load(handle)
    preset["_path"] = str(path)
    return preset


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def resolve_user_path(value: str) -> Path:
    """Interpret paths accepted by the root CLI relative to the repository."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def last_option_value(command: list[str], option: str) -> Optional[str]:
    """Return the last value of an option so appended overrides are respected."""
    value = None
    for index, item in enumerate(command[:-1]):
        if item == option:
            value = command[index + 1]
    return value


def run_command(command: list[str], cwd: Path, dry_run: bool = False) -> int:
    print(f"[cwd] {cwd}")
    print(f"[command] {format_command(command)}")
    if dry_run:
        return 0
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    return int(completed.returncode)


def command_doctor(_: argparse.Namespace) -> int:
    failures = []
    print(f"repository: {REPO_ROOT}")
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 9):
        failures.append("recommended Python version is 3.9")

    print("\nTraining dependencies:")
    for module, package in TRAIN_IMPORTS.items():
        available = importlib.util.find_spec(module) is not None
        print(f"  [{'ok' if available else 'missing'}] {package}")
        if not available:
            failures.append(f"missing Python package: {package}")

    resources = {
        "dual-objective reward cache": DATASET_ROOT
        / "offline/bert_ec_final_offline_mic_priority_top2000_mic_hemo_bert_ec_final_mic_priority_motif67_rewards.pt",
        "three-objective reward cache": DATASET_ROOT
        / "offline/bert_ec_final_offline_mic_priority_top2000_mic_hemo_tox_bert_ec_final_mic_hemo_tox_motif96_rewards.pt",
        "BERT MIC service": MIC_SERVER,
        "BERT MIC backbone": PREDICTOR_ROOT
        / "bert_mic/artifacts/backbone/pytorch_model.bin",
        "BERT MIC checkpoint": PREDICTOR_ROOT
        / "bert_mic/artifacts/best_model_state_dict.pkl",
        "BERT MIC calibration": PREDICTOR_ROOT
        / "bert_mic/artifacts/score_calibration.json",
    }
    minimum_resource_sizes = {
        "BERT MIC service": 1_000,
        "BERT MIC backbone": 1_000_000_000,
        "BERT MIC checkpoint": 1_000_000_000,
        "BERT MIC calibration": 50,
    }
    print("\nProject resources:")
    for label, path in resources.items():
        exists = path.is_file()
        complete = exists and path.stat().st_size >= minimum_resource_sizes.get(label, 1)
        print(f"  [{'ok' if complete else 'missing/incomplete'}] {label}: {path}")
        if not complete:
            failures.append(f"missing or incomplete resource: {label}")

    hemo_model = os.environ.get("HEMO_MODEL_DIR")
    toxinpred = os.environ.get("TOXINPRED3_BIN")
    print("\nOptional local reward-service resources:")
    print(f"  HEMO_MODEL_DIR={hemo_model or '<not set>'}")
    print(f"  TOXINPRED3_BIN={toxinpred or '<not set>'}")

    if failures:
        print("\nDoctor result: NOT READY")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nDoctor result: READY")
    return 0


def post_scores(
    url: str,
    payload: dict,
    key: str,
    label: str,
    probability: bool = True,
) -> float:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} service failed at {url}: {exc}") from exc

    values = body.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(f"{label} must return one value in {key!r}; response={body}")
    score = float(values[0])
    if not math.isfinite(score):
        raise RuntimeError(f"{label} returned a non-finite score: {score}")
    if probability and not 0.0 <= score <= 1.0:
        raise RuntimeError(f"{label} returned {score}; expected a value in [0, 1]")
    return score


def check_selected_oracles(preset: dict, sequence: str) -> dict[str, float]:
    objectives = preset["objectives"]
    mic_backend = preset.get("mic_oracle", "bert")
    results: dict[str, float] = {}

    if "mic" in objectives:
        if mic_backend == "bert":
            url = os.environ.get(
                "MIC_BERT_URL",
                f"http://127.0.0.1:{os.environ.get('MIC_BERT_PORT', '5010')}/predict",
            )
            results["mic"] = post_scores(url, {"seqs": [sequence]}, "mic_score", "BERT MIC")
        else:
            url = os.environ.get(
                "MBC_MIC_URL",
                f"http://127.0.0.1:{os.environ.get('MBC_MIC_PORT', '5011')}/predict",
            )
            results["mic_pmic"] = post_scores(
                url, {"seqs": [sequence]}, "pmic", "MBC MIC", probability=False
            )

    if "hemo" in objectives:
        url = os.environ.get(
            "HEMO_URL",
            f"http://127.0.0.1:{os.environ.get('HEMO_PORT', '5006')}/predict",
        )
        results["hemo_probability"] = post_scores(
            url, {"sequences": [sequence]}, "scores", "HemoPI2"
        )

    if "tox" in objectives:
        url = os.environ.get(
            "TOX_URL",
            f"http://127.0.0.1:{os.environ.get('TOX_PORT', '5008')}/predict",
        )
        results["tox_probability"] = post_scores(
            url, {"sequences": [sequence]}, "scores", "ToxinPred3"
        )

    if "amp" in objectives:
        url = os.environ.get(
            "AMP_URL",
            f"http://127.0.0.1:{os.environ.get('AMP_PORT', '5007')}/predict",
        )
        results["amp"] = post_scores(url, {"sequences": [sequence]}, "scores", "AMP")

    return results


def command_check(args: argparse.Namespace) -> int:
    preset = load_preset(args.preset)
    try:
        results = check_selected_oracles(preset, args.sequence)
    except RuntimeError as exc:
        print(f"Reward-service check failed: {exc}", file=sys.stderr)
        return 1
    print(f"Reward-service check passed: {results}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    targets = {
        "mic": MIC_SERVER,
        "hemo": PREDICTOR_ROOT / "hemolysis" / "server.py",
        "tox": PREDICTOR_ROOT / "toxicity" / "server.py",
    }
    target = targets[args.service]
    if not target.is_file():
        print(f"Service entry does not exist: {target}", file=sys.stderr)
        return 1
    python = args.python or sys.executable
    return run_command([python, str(target)], target.parent, dry_run=args.dry_run)


def command_train(args: argparse.Namespace) -> int:
    preset = load_preset(args.preset)
    command = [
        sys.executable,
        "-u",
        "-m",
        "gflownet.tasks.amp_moo",
        *preset["train_args"],
    ]
    if args.steps is not None:
        command.extend(["--num_training_steps", str(args.steps)])
    if args.log_dir:
        command.extend(["--log_dir", str(resolve_user_path(args.log_dir))])
    if args.skip_oracle_preflight:
        command.append("--skip_oracle_preflight")
    if args.extra:
        # argparse.REMAINDER keeps the conventional separator. It is for this
        # wrapper only and must not be forwarded to amp_moo itself.
        extra = args.extra[1:] if args.extra[0] == "--" else args.extra
        command.extend(extra)

    log_dir_value = last_option_value(command, "--log_dir")
    if log_dir_value:
        log_dir = Path(log_dir_value)
        if not log_dir.is_absolute():
            log_dir = (REPO_ROOT / log_dir).resolve()
        if log_dir.exists() and not args.overwrite and not args.dry_run:
            print(
                f"Training output already exists: {log_dir}\n"
                "Choose another --log-dir, or pass --overwrite to replace it.",
                file=sys.stderr,
            )
            return 2
    return run_command(command, REPO_ROOT, dry_run=args.dry_run)


def command_sample(args: argparse.Namespace) -> int:
    preset = load_preset(args.preset)
    sampling = preset["sampling"]
    checkpoint = resolve_user_path(args.checkpoint)
    output = resolve_user_path(args.output)
    if not args.dry_run and not checkpoint.is_file():
        print(f"Checkpoint does not exist: {checkpoint}", file=sys.stderr)
        return 2
    if output.exists() and not args.overwrite and not args.dry_run:
        print(
            f"Sampling output already exists: {output}\n"
            "Choose another --output, or pass --overwrite to replace it.",
            file=sys.stderr,
        )
        return 2
    command = [
        sys.executable,
        str(SCRIPTS_ROOT / "sample_trained_amp.py"),
        "--checkpoint",
        str(checkpoint),
        "--motif_vocab_path",
        sampling["motif_vocab_path"],
        "--output_csv",
        str(output),
        "--sampling_seed",
        str(args.seed),
        "--num_samples",
        str(args.num_samples),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--weights",
        args.weights,
        "--mic_oracle",
        preset.get("mic_oracle", "bert"),
        "--max_motif_actions",
        str(sampling.get("max_motif_actions", 2)),
        "--min_length",
        str(sampling.get("min_length", 5)),
        "--motif_use_gate_strength",
        str(sampling.get("motif_use_gate_strength", 1.0)),
        "--motif_select_gate_strength",
        str(sampling.get("motif_select_gate_strength", 1.0)),
    ]
    if sampling.get("use_motif_gate", False):
        command.append("--use_motif_gate")
    if args.allow_legacy_checkpoint:
        command.append("--allow_legacy_checkpoint")
    return run_command(command, REPO_ROOT, dry_run=args.dry_run)


def command_test(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]
    return run_command(command, REPO_ROOT, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry point for MiOP-GFN training and reward services."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check the local environment and resources")
    doctor.set_defaults(func=command_doctor)

    check = subparsers.add_parser("check", help="send one sequence to selected reward services")
    check.add_argument("--preset", default="mic-hemo", choices=["mic-hemo", "mic-hemo-tox"])
    check.add_argument("--sequence", default="KIVRIFFKILKF")
    check.set_defaults(func=command_check)

    serve = subparsers.add_parser("serve", help="start one local reward service")
    serve.add_argument("service", choices=["mic", "hemo", "tox"])
    serve.add_argument("--python", help="Python executable for this service environment")
    serve.add_argument("--dry-run", action="store_true")
    serve.set_defaults(func=command_serve)

    train = subparsers.add_parser("train", help="train from a named preset")
    train.add_argument("--preset", default="mic-hemo", choices=["mic-hemo", "mic-hemo-tox"])
    train.add_argument("--steps", type=int, help="override num_training_steps")
    train.add_argument("--log-dir", help="override the preset log directory")
    train.add_argument(
        "--overwrite",
        action="store_true",
        help="allow amp_moo to replace an existing training output directory",
    )
    train.add_argument("--skip-oracle-preflight", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("extra", nargs=argparse.REMAINDER, help="extra amp_moo args after --")
    train.set_defaults(func=command_train)

    sample = subparsers.add_parser("sample", help="sample sequences from a checkpoint")
    sample.add_argument("--preset", default="mic-hemo", choices=["mic-hemo", "mic-hemo-tox"])
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--num-samples", type=int, default=5000)
    sample.add_argument("--batch-size", type=int, default=64)
    sample.add_argument("--device", default="cuda")
    sample.add_argument("--weights", choices=["auto", "ema", "model"], default="auto")
    sample.add_argument("--allow-legacy-checkpoint", action="store_true")
    sample.add_argument("--overwrite", action="store_true", help="replace an existing CSV")
    sample.add_argument("--dry-run", action="store_true")
    sample.set_defaults(func=command_sample)

    test = subparsers.add_parser("test", help="run the repository unit tests")
    test.add_argument("--dry-run", action="store_true")
    test.set_defaults(func=command_test)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
