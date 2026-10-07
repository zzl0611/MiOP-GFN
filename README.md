# MiOP-GFN

MiOP-GFN is a multi-objective GFlowNet framework for antimicrobial peptide generation. It combines order-preserving trajectory balance, offline peptide data, online reward prediction, and motif-guided sequence construction to generate diverse peptide candidates under multiple biological objectives.

The current repository supports:

- dual-objective generation for antibacterial activity and low hemolysis;
- three-objective generation for antibacterial activity, low hemolysis, and low toxicity;
- mixed offline/online training with replay;
- motif-aware sequence actions and motif gating;
- Pareto-oriented multi-objective optimization;
- reproducible checkpoint sampling with motif and configuration validation.

This repository focuses on model training and raw sequence generation. Independent downstream evaluation pipelines are not required for training.

## Contents

- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reward predictors](#reward-predictors)
- [Datasets and motif vocabularies](#datasets-and-motif-vocabularies)
- [Training](#training)
- [Sampling](#sampling)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Testing and troubleshooting](#testing-and-troubleshooting)
- [Citation and license](#citation-and-license)

## Method overview

MiOP-GFN generates peptide sequences from amino-acid and motif actions. During training, every generated sequence is assigned a vector of objective rewards, and the order-preserving GFlowNet objective learns to sample candidates according to their multi-objective quality.

```text
Offline FASTA + cached rewards
              │
              ├──────────────┐
              ▼              ▼
       offline samples   online generation
                              │
                              ▼
                    reward predictors (HTTP)
                    ├── BERT MIC
                    ├── HemoPI2
                    └── ToxinPred3 (optional)
                              │
                              ▼
                  multi-objective reward vector
                              │
                              ▼
               OP-TB training + replay + motif gate
                              │
                              ▼
                    checkpoint and Pareto records
```

The optimization direction of every configured objective is “higher is better”:

| Objective | Predictor output used by MiOP-GFN | Training reward |
| --- | --- | --- |
| `mic` | calibrated BERT MIC score | higher indicates stronger predicted antibacterial activity |
| `hemo` | hemolysis probability | `1 - probability`, so higher indicates lower predicted hemolysis |
| `tox` | toxicity probability | `1 - probability`, so higher indicates lower predicted toxicity |

The paper-aligned `mic-hemo` configuration uses the OP-TB training objective without offline STOP loss, online STOP loss, or motif-diversity loss.

## Repository structure

```text
MiOP-GFN/
├── run.py                         # unified command-line entry point
├── configs/
│   ├── README.md
│   └── train/
│       ├── mic-hemo.json          # MIC + hemolysis preset
│       └── mic-hemo-tox.json      # MIC + hemolysis + toxicity preset
├── datasets/
│   ├── README.md
│   ├── offline/                   # offline FASTA and reward caches
│   └── motifs/                    # motif vocabularies
├── predictors/
│   ├── README.md
│   ├── bert_mic/
│   │   ├── server.py              # BERT MIC HTTP service
│   │   ├── regressor.py           # MIC regression architecture
│   │   └── artifacts/             # backbone, checkpoint, calibration
│   ├── hemolysis/
│   │   └── server.py              # HemoPI2 service adapter
│   └── toxicity/
│       └── server.py              # ToxinPred3 service adapter
├── src/gflownet/
│   ├── algo/                      # GFlowNet algorithms and OP-TB
│   ├── data/                      # replay buffer and sampling iterator
│   ├── envs/                      # sequence and graph environments
│   ├── models/                    # MiOP-GFN model implementations
│   ├── tasks/                     # AMP task and reward clients
│   └── utils/                     # metrics, conditioning, logging, manifests
├── scripts/
│   └── sample_trained_amp.py      # checkpoint sampling implementation
├── tests/                         # unit and reproducibility tests
├── docs/                          # detailed running and architecture guides
├── outputs/                       # generated at runtime; ignored by Git
├── environment-amp.yml
└── LICENSE.md
```

The `src/gflownet/models/` directory contains the trainable MiOP-GFN model code. Reward-predictor implementations and their runtime resources are kept separately in `predictors/`.

## Installation

### Requirements

- Python 3.9;
- Conda or Miniconda;
- Git LFS for the BERT MIC weights;
- CUDA-capable PyTorch for GPU training, or CPU PyTorch for code-level tests;
- accessible HemoPI2 and optional ToxinPred3 resources or remote services.

### Create the environment

From the repository root:

```bash
git lfs install
git lfs pull
conda env create -f environment-amp.yml
conda activate amp-moo-gfn
python run.py doctor
```

The two large BERT MIC weight files are managed with Git LFS. `run.py doctor` checks both existence and approximate file size, so an unresolved LFS pointer is reported as an incomplete resource.

`environment-amp.yml` provides a CPU-compatible base environment. For GPU training, install the PyTorch/CUDA build matching the server driver instead of the `cpuonly` package.

## Reward predictors

MiOP-GFN communicates with predictors through HTTP. Predictor failures are fatal by default: the training code does not silently replace missing predictions with fabricated rewards.

| Predictor | Default endpoint | Required for | Local resource status |
| --- | --- | --- | --- |
| BERT MIC | `http://127.0.0.1:5010/predict` | all provided presets | code and weights included |
| HemoPI2 | `http://127.0.0.1:5006/predict` | `mic-hemo`, `mic-hemo-tox` | adapter included; model supplied separately |
| ToxinPred3 | `http://127.0.0.1:5008/predict` | `mic-hemo-tox` | adapter included; executable supplied separately |

### Option 1: use remote predictor services

Linux/macOS:

```bash
export MIC_BERT_URL=http://SERVER:5010/predict
export HEMO_URL=http://SERVER:5006/predict
export TOX_URL=http://SERVER:5008/predict  # three-objective training only
```

PowerShell:

```powershell
$env:MIC_BERT_URL = 'http://SERVER:5010/predict'
$env:HEMO_URL = 'http://SERVER:5006/predict'
$env:TOX_URL = 'http://SERVER:5008/predict'  # three-objective training only
```

### Option 2: run predictors locally

Start each required service in a separate terminal:

```powershell
# BERT MIC: all resources are under predictors/bert_mic/
python .\run.py serve mic

# HemoPI2: point to the separately obtained model directory
$env:HEMO_MODEL_DIR = 'D:\path\to\hemopi2\Model'
python .\run.py serve hemo --python 'D:\path\to\hemo-env\python.exe'

# ToxinPred3: required only by mic-hemo-tox
$env:TOXINPRED3_BIN = 'D:\path\to\toxinpred3.exe'
python .\run.py serve tox --python 'D:\path\to\tox-env\python.exe'
```

Before training, send a real peptide through every predictor selected by the preset:

```bash
python run.py check --preset mic-hemo
python run.py check --preset mic-hemo-tox
```

See [docs/LOCAL_REWARD_SERVICES.md](docs/LOCAL_REWARD_SERVICES.md) for request/response contracts, port overrides, and local service requirements.

## Datasets and motif vocabularies

The repository contains the resources used by the retained training presets:

| Resource | Location | Purpose |
| --- | --- | --- |
| offline peptide sequences | `datasets/offline/*.fasta` | offline component of mixed training |
| cached reward tensors | `datasets/offline/*_rewards.pt` | avoids recomputing rewards for unchanged offline data |
| dual-objective motif vocabulary | `datasets/motifs/*mic_hemo_notox*.txt` | motif actions for `mic-hemo` |
| three-objective motif vocabulary | `datasets/motifs/*mic_hemo_tox*.txt` | motif actions for `mic-hemo-tox` |

Reward-cache filenames depend on the FASTA stem, objective order, and `--reward_cache_tag`. Changing any of these values selects a different cache and may require the predictor services to recompute offline rewards.

Motif order determines action indices. Training and sampling must use the same vocabulary. New checkpoints store a reproducibility manifest that validates the motif file hash, action mapping, gate settings, and length-related parameters.

More details are available in [datasets/README.md](datasets/README.md).

## Training

All recommended commands use the root entry point `run.py`. It adds `src/` to `PYTHONPATH`, resolves paths from the repository root, and prevents accidental reuse of an existing output directory.

### Available presets

| Preset | Objectives | MIC predictor | Default steps | Default output |
| --- | --- | --- | ---: | --- |
| `mic-hemo` | MIC, low hemolysis | BERT MIC | 10,000 | `outputs/training/paper_mic_hemo_10000_off020_rand015_seed0/` |
| `mic-hemo-tox` | MIC, low hemolysis, low toxicity | BERT MIC | 10,000 | `outputs/training/mic-hemo-tox/` |

The main `mic-hemo` preset uses:

| Parameter | Value |
| --- | ---: |
| global batch size | 64 |
| offline ratio | 0.20 |
| offline reward batch size | 64 |
| random action probability | 0.15 |
| maximum motif actions | 2 |
| motif-use gate strength | 1.0 |
| motif-selection gate strength | 1.0 |
| motif-usage loss weight | 0.0 |
| Pareto length limit | 30 |
| replay length limit | 30 |

### Inspect the resolved command

```bash
python run.py train --preset mic-hemo --dry-run
```

### Smoke test

Run a short experiment in a separate output directory before a full run:

```bash
python run.py train \
  --preset mic-hemo \
  --steps 10 \
  --log-dir ./outputs/training/smoke-mic-hemo
```

### Full dual-objective training

```bash
python run.py train --preset mic-hemo
```

### Full three-objective training

```bash
python run.py train --preset mic-hemo-tox
```

### Override selected parameters

Arguments after `--` are passed to `gflownet.tasks.amp_moo`; later values override the preset:

```bash
python run.py train \
  --preset mic-hemo \
  --log-dir ./outputs/training/mic-hemo-seed42 \
  -- --seed 42 --num_training_steps 20000
```

For a permanent experiment, copy a JSON file under `configs/train/` and record a distinct output directory instead of relying only on shell history.

## Sampling

Sampling restores the trained generator and writes raw peptide candidates. It does not call the reward predictors or perform downstream biological evaluation.

```powershell
python .\run.py sample `
  --preset mic-hemo `
  --checkpoint .\outputs\training\paper_mic_hemo_10000_off020_rand015_seed0\model_state.pt `
  --output .\outputs\samples\mic-hemo-seed0.csv `
  --seed 0 `
  --num-samples 5000 `
  --batch-size 64 `
  --device cuda
```

By default, sampling uses the EMA sampling weights when present. The `--weights` option accepts `auto`, `ema`, or `model`. Checkpoints created before the reproducibility manifest was added require the explicit `--allow-legacy-checkpoint` flag.

## Outputs

A training directory normally contains:

```text
outputs/training/EXPERIMENT/
├── hps.yaml                     # resolved training configuration
├── model_state.pt               # model, EMA, optimizer, and manifest state
├── pareto.pt                    # accumulated Pareto records
├── train.log                    # training log
├── events.out.tfevents.*        # TensorBoard events
├── train/                       # generated training samples
├── valid/                       # validation samples
└── final/                       # final-generation samples
```

Do not run two experiments with the same `log_dir`. `run.py` stops when the target already exists unless `--overwrite` is explicitly supplied.

## Reproducibility

For a reproducible run:

1. keep the preset JSON and exact Git revision;
2. retain the motif vocabulary without reordering lines;
3. record the reward-service model versions and endpoint configuration;
4. use a unique output directory for every seed;
5. archive `hps.yaml`, `model_state.pt`, and the reward cache together;
6. do not enable `ORACLE_ALLOW_FALLBACK=1` in real experiments.

The checkpoint manifest validates critical sampling state, including motif SHA-256, action mapping, gate strengths, minimum length, and STOP/action settings.

## Testing and troubleshooting

Run the repository tests without contacting predictor services:

```bash
python run.py test
```

Useful diagnostic commands:

```bash
python run.py doctor
python run.py check --preset mic-hemo
python run.py train --preset mic-hemo --dry-run
```

Common problems:

- **BERT files are only a few bytes or kilobytes:** run `git lfs pull`.
- **`doctor` reports missing packages:** activate the `amp-moo-gfn` environment and verify Python 3.9.
- **Predictor connection is refused:** confirm the service is running and that the complete `/predict` URL is configured.
- **HemoPI2 model is missing:** set `HEMO_MODEL_DIR` to its Hugging Face-style model directory.
- **ToxinPred3 executable is missing:** set `TOXINPRED3_BIN` to the executable path.
- **Training fails after skipping preflight:** `--skip-oracle-preflight` does not remove the need for online rewards once new sequences are generated.

Additional documentation:

- [Project structure](docs/PROJECT_STRUCTURE.md)
- [Running guide](docs/RUNNING.md)
- [Reward predictor services](docs/LOCAL_REWARD_SERVICES.md)
- [Training configurations](configs/README.md)
- [Dataset resources](datasets/README.md)
- [Exact paper-aligned manifest](docs/AMP_MOO_EXACT_MANIFEST.md)

## Citation and license

MiOP-GFN builds on Order-Preserving GFlowNets:

```bibtex
@inproceedings{chen2024orderpreserving,
  title={Order-Preserving {GF}lowNets},
  author={Yihang Chen and Lukas Mauch},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=VXDPXuq4oG}
}
```

Repository code and documentation are distributed under the terms in [LICENSE.md](LICENSE.md). Third-party predictor resources may be subject to their original licenses or model-card terms; see `predictors/bert_mic/artifacts/backbone/MODEL_CARD.md` for the bundled backbone information.
