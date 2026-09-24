# Exact AMP-MOO Manifest for the 0715 MIC+Hemo Command

This manifest is for this exact training command, run from `multi/`:

```bash
MIC_BERT_PORT=5010 HEMO_PORT=5006 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONUNBUFFERED=1 \
python -u -m gflownet.tasks.amp_moo \
  --log_dir ./logs/0715_bert_motif97_10000_off020_rand015_usage0.0_seed0 \
  --objectives mic hemo \
  --mic_oracle bert \
  --type ordering \
  --compute_hvi \
  --compute_igd \
  --compute_pc_entropy \
  --replay \
  --seed 0 \
  --num_training_steps 10000 \
  --offline_data ./motif_discovery/data/bert_ec_final_offline_mic_priority_top2000.fasta \
  --offline_ratio 0.20 \
  --offline_reward_batch_size 64 \
  --reward_cache_tag bert_ec_final_mic_priority_motif67 \
  --motif_vocab_path ./motif_discovery/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt \
  --max_motif_actions 2 \
  --use_motif_gate \
  --motif_use_gate_strength 1.0 \
  --motif_select_gate_strength 1.0 \
  --motif_usage_loss_weight 0.05 \
  --motif_diversity_loss_weight 0.05 \
  --offline_stop_loss_weight 0.10 \
  --online_stop_len 24 \
  --online_stop_loss_weight 0.10 \
  --pareto_len_limit 35 \
  --replay_len_limit 30 \
  --train_random_action_prob 0.15
```

## Files Directly Used by This Command

Entry and AMP sequence implementation:

- `gflownet/tasks/amp_moo.py`
- `gflownet/tasks/oracle.py`
- `gflownet/envs/seq_env.py`
- `gflownet/models/seq_model.py`
- `gflownet/algo/seq_trajectory_balance.py`
- `gflownet/data/sampling_iterator.py`
- `gflownet/data/replay_buffer.py`
- `gflownet/utils/multiobjective_hooks.py`
- `gflownet/utils/metrics.py`
- `gflownet/utils/conditioning.py`
- `gflownet/utils/misc.py`
- `gflownet/utils/multiprocessing_proxy.py`
- `gflownet/utils/reproducibility.py`
- `gflownet/config.py`
- `gflownet/algo/config.py`
- `gflownet/data/config.py`
- `gflownet/models/config.py`
- `gflownet/tasks/config.py`
- `gflownet/utils/config.py`
- `gflownet/trainer.py`
- `gflownet/online_trainer.py`

Resources:

- `motif_discovery/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt`
- `motif_discovery/data/bert_ec_final_offline_mic_priority_top2000.fasta`
- `motif_discovery/data/bert_ec_final_offline_mic_priority_top2000_mic_hemo_bert_ec_final_mic_priority_motif67_rewards.pt`

Services:

- BERT-MIC service: `http://127.0.0.1:5010/predict`, expected output key `mic_score`.
- Hemo service: `http://127.0.0.1:5006/predict`, expected output key `scores`.

Output directory created by the command:

- `logs/0715_bert_motif97_10000_off020_rand015_usage0.0_seed0/`

## Files Currently Pulled In Only Because of Inheritance/Top-Level Imports

These are not conceptually part of AMP-MOO, but they are currently imported
when `amp_moo.py` imports `SEHMOOTask` and `StandardOnlineTrainer`:

- `gflownet/tasks/seh_frag_moo.py`
- `gflownet/tasks/seh_frag.py`
- `gflownet/envs/graph_building_env.py`
- `gflownet/envs/frag_mol_env.py`
- `gflownet/models/graph_transformer.py`
- `gflownet/models/bengio2021flow.py`
- `gflownet/algo/trajectory_balance.py`
- `gflownet/algo/flow_matching.py`
- `gflownet/algo/advantage_actor_critic.py`
- `gflownet/algo/soft_q_learning.py`
- `gflownet/algo/envelope_q_learning.py`
- `gflownet/algo/multiobjective_reinforce.py`
- `gflownet/algo/graph_sampling.py`
- `gflownet/utils/sascore.py`
- `gflownet/utils/transforms.py`
- `gflownet/utils/focus_model.py`

They can be deleted only after refactoring `amp_moo.py` away from
`SEHMOOTask` and `StandardOnlineTrainer` or after editing those base modules so
their graph/molecule imports are lazy.

## Why Keeping Only SEHMOOTask and StandardOnlineTrainer Is Not Enough

The issue is not only class inheritance. It is Python import execution.

`amp_moo.py` imports:

- `from gflownet.tasks.seh_frag_moo import SEHMOOTask`
- `from gflownet.online_trainer import StandardOnlineTrainer`
- `from gflownet.trainer import FlatRewards`

Then:

- `seh_frag_moo.py` imports RDKit descriptors, fragment molecule environments,
  graph algorithms, graph models, and `SEHTask`.
- `online_trainer.py` imports all original graph algorithms and
  `GraphTransformerGFN`.
- `trainer.py` imports graph batch/environment classes and RDKit logging.

So keeping only those two classes in their current files still requires many
non-AMP files to exist and many non-AMP dependencies to be installable.

## What AMP-MOO Actually Needs from SEHMOOTask

For the command above, AMP-MOO uses only the conditioning and scalarization
parts:

- temperature conditional sampling;
- Dirichlet preference sampling;
- conditional encoding size;
- `flat_reward_transform`;
- `sample_conditional_information`;
- `encode_conditional_information` for validation;
- `cond_info_to_logreward`.

These can be moved into a small sequence-only task base, for example
`gflownet/tasks/amp_conditioning.py`, using only:

- `torch`
- `numpy`
- `gflownet.utils.conditioning.MultiObjectiveWeightedPreferences`
- `gflownet.utils.conditioning.TemperatureConditional`

## What AMP-MOO Actually Needs from StandardOnlineTrainer

For this command, AMP-MOO uses:

- config merge through `GFNTrainer.__init__`;
- setup order;
- replay buffer creation, especially `MOOReplayBuffer`;
- Adam optimizers for model and `logZ`;
- EMA sampling model controlled by `cfg.algo.sampling_tau`;
- gradient clipping;
- checkpoint saving;
- training/validation/final-generation loops.

These can be moved into a sequence-only trainer, for example
`gflownet/amp_trainer.py`, using only:

- `gflownet.data.sampling_iterator.SamplingIterator`
- `gflownet.data.replay_buffer.MOOReplayBuffer`
- `gflownet.config.Config`
- `gflownet.utils.misc.create_logger`

That refactor would allow deleting almost all graph/molecule files from the
public AMP-MOO repository.

## Cleanup Applied

The following unrelated or generated paths have been removed from this working
tree because the exact AMP training and sampling paths do not use them:

- `multi/hypergrid_comb.py`
- `multi/grid_cond_gfn.py`
- `multi/gflownet/tasks/qm9/`
- `multi/gflownet/tasks/eval_offline_moo.py`
- `multi/gflownet/tasks/make_rings.py`
- `multi/gflownet/tasks/sampling.py`
- `multi/analysis_results/`
- `multi/results/`
- `multi/formal_samples/`
- `multi/sampled_*`
- `multi/analysis_code/`
- `multi/amp_oracle/`
- `multi/MIC_oracle/`
- `multi/logs/`
- backup scripts ending in `_backup.py` or `_before_singleton_fix.py`
- copied exploratory scripts such as `bpe_motif_mic_hemo_tox copy.py`

The remaining original graph/molecule modules must not be deleted until either:

1. `amp_moo.py` no longer imports `SEHMOOTask`, `StandardOnlineTrainer`, or
   `FlatRewards` from graph-coupled files; or
2. the top-level imports in those modules are made lazy and verified in a clean
   environment.
