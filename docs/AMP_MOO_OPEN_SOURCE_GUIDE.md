# AMP-MOO Open-Source Guide

This repository contains the AMP multi-objective optimization task derived from
Order-Preserving GFlowNets. The supported entry points are:

- training: `multi/gflownet/tasks/amp_moo.py`
- checkpoint sampling: `multi/sample_trained_amp.py`
- hemolysis service helper: `multi/hemo_server.py`
- toxicity service helper: `multi/tox_server.py`

Historical experiment scripts, checkpoints, logs, generated samples, analysis
outputs, motif-discovery pipelines, and unused oracle implementations have been
removed from this release tree.

## Included runtime resources

The repository retains only the data used by the final MIC+hemo and
MIC+hemo+tox configurations:

- `multi/motif_discovery/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt`
- `multi/motif_discovery/bpe_motif_vocab_bert_ec_final_mic_hemo_tox_effect005.txt`
- `multi/motif_discovery/bpe_motif_vocab_final.txt` (the command-line default)
- `multi/motif_discovery/data/bert_ec_final_offline_mic_priority_top2000.fasta`
- the matching dual-objective reward cache
- the matching three-objective reward cache

Reward-cache filenames are derived from the FASTA stem, objective order, and
`--reward_cache_tag`. Changing any of these selects a different cache and may
require the oracle services to recompute rewards.

## Environment

Create the AMP environment from the repository root:

```bash
conda env create -f environment-amp.yml
conda activate amp-moo-gfn
cd multi
```

Graph/molecule modules remain in `multi/gflownet` because the current AMP task
inherits the original multi-objective trainer and its eager imports. They are
runtime dependencies even though the AMP experiment does not generate
molecules.

## Oracle endpoints

The objective wrappers in `multi/gflownet/tasks/oracle.py` use these local HTTP
services:

- BERT MIC: `MIC_BERT_URL`, or port `MIC_BERT_PORT` (default `5010`)
- MBC MIC: `MBC_MIC_URL`, or port `MBC_MIC_PORT` (default `5011`)
- hemolysis: port `HEMO_PORT` (default `5006`)
- toxicity: `TOX_URL`, or port `TOX_PORT` (default `5008`)

The BERT-MIC service/model is external to this repository. The included hemo
and tox helpers still require their separately obtained model/tool resources.

## Dual-objective training

Run from `multi/`. This is the retained MIC+hemo experiment configuration:

```bash
MIC_BERT_PORT=5010 HEMO_PORT=5006 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONUNBUFFERED=1 \
python -u -m gflownet.tasks.amp_moo \
  --log_dir ./logs/0715_bert_motif97_10000_off020_rand015_usage0.0_seed0 \
  --objectives mic hemo \
  --mic_oracle bert \
  --type ordering \
  --compute_hvi --compute_igd --compute_pc_entropy \
  --replay --seed 0 --num_training_steps 10000 \
  --offline_data ./motif_discovery/data/bert_ec_final_offline_mic_priority_top2000.fasta \
  --offline_ratio 0.20 --offline_reward_batch_size 64 \
  --reward_cache_tag bert_ec_final_mic_priority_motif67 \
  --motif_vocab_path ./motif_discovery/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt \
  --max_motif_actions 2 --use_motif_gate \
  --motif_use_gate_strength 1.0 --motif_select_gate_strength 1.0 \
  --motif_usage_loss_weight 0.05 --motif_diversity_loss_weight 0.05 \
  --offline_stop_loss_weight 0.10 --online_stop_len 24 \
  --online_stop_loss_weight 0.10 --pareto_len_limit 35 \
  --replay_len_limit 30 --train_random_action_prob 0.15
```

The command creates `logs/` when it starts; output directories are intentionally
not versioned.

## Sampling

New checkpoints contain a `reproducibility_manifest`. Sampling validates the
motif SHA-256, motif/action mapping, gate configuration, and STOP/length
settings before loading weights:

```bash
python sample_trained_amp.py \
  --checkpoint ./logs/EXPERIMENT/model_state.pt \
  --motif_vocab_path ./motif_discovery/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt \
  --output_csv ./samples.csv \
  --sampling_seed 0 \
  --num_samples 5000 \
  --batch_size 64 \
  --device cuda \
  --weights auto \
  --mic_oracle bert \
  --max_motif_actions 2 \
  --min_length 5 \
  --use_motif_gate \
  --motif_use_gate_strength 1.0 \
  --motif_select_gate_strength 1.0
```

Old checkpoints created before the manifest change require
`--allow_legacy_checkpoint`. That compatibility switch cannot prove that motif
ordering, gate strengths, or length settings match the original training run.

## Verification

The manifest unit tests do not contact oracle services:

```bash
cd multi
python -m unittest discover -s tests -p "test_*.py" -v
```

For the exact dual-objective hashes and parameter record, see
`docs/AMP_MOO_EXACT_MANIFEST.md`.
