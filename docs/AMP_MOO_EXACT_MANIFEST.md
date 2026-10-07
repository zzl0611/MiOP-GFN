# Paper-Aligned AMP-MOO Manifest for MIC+Hemo

This manifest records the paper-aligned training command, run from the repository root.
Unlike the historical 0627 command, it contains no offline STOP loss, online
STOP loss, or motif-diversity loss:

```bash
MIC_BERT_PORT=5010 HEMO_PORT=5006 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./src PYTHONUNBUFFERED=1 \
python -u -m gflownet.tasks.amp_moo \
  --log_dir ./outputs/training/paper_mic_hemo_10000_off020_rand015_seed0 \
  --objectives mic hemo \
  --mic_oracle bert \
  --type ordering \
  --compute_hvi \
  --compute_igd \
  --compute_pc_entropy \
  --replay \
  --seed 0 \
  --num_training_steps 10000 \
  --offline_data ./datasets/offline/bert_ec_final_offline_mic_priority_top2000.fasta \
  --offline_ratio 0.20 \
  --offline_reward_batch_size 64 \
  --reward_cache_tag bert_ec_final_mic_priority_motif67 \
  --motif_vocab_path ./datasets/motifs/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt \
  --max_motif_actions 2 \
  --use_motif_gate \
  --motif_use_gate_strength 1.0 \
  --motif_select_gate_strength 1.0 \
  --motif_usage_loss_weight 0.0 \
  --pareto_len_limit 30 \
  --replay_len_limit 30 \
  --train_random_action_prob 0.15
```

The defaults in `src/gflownet/tasks/amp_moo.py` now match this command. Therefore,
from the repository root, `PYTHONPATH=./src python -u -m gflownet.tasks.amp_moo` selects the same training
configuration. The explicit command remains the clearest archival record.

The historical code used `global_batch_size=64`. This is distinct from the
command's `offline_reward_batch_size=64`: the former controls the mixed
offline/online training batch, while the latter only controls chunking when an
offline reward cache must be computed. The training default is explicitly kept
at 64 and can now be overridden with `--global_batch_size`.

## Files Directly Used by This Command

Entry and AMP sequence implementation:

- `src/gflownet/tasks/amp_moo.py`
- `src/gflownet/tasks/oracle.py`
- `src/gflownet/envs/seq_env.py`
- `src/gflownet/models/seq_model.py`
- `src/gflownet/algo/seq_trajectory_balance.py`
- `src/gflownet/data/sampling_iterator.py`
- `src/gflownet/data/replay_buffer.py`
- `src/gflownet/utils/multiobjective_hooks.py`
- `src/gflownet/utils/metrics.py`
- `src/gflownet/utils/conditioning.py`
- `src/gflownet/utils/misc.py`
- `src/gflownet/utils/multiprocessing_proxy.py`
- `src/gflownet/utils/reproducibility.py`
- `src/gflownet/config.py`
- `src/gflownet/algo/config.py`
- `src/gflownet/data/config.py`
- `src/gflownet/models/config.py`
- `src/gflownet/tasks/config.py`
- `src/gflownet/utils/config.py`
- `src/gflownet/trainer.py`
- `src/gflownet/online_trainer.py`

Resources:

- `datasets/motifs/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt`
- `datasets/offline/bert_ec_final_offline_mic_priority_top2000.fasta`
- `datasets/offline/bert_ec_final_offline_mic_priority_top2000_mic_hemo_bert_ec_final_mic_priority_motif67_rewards.pt`

Services:

- BERT-MIC service: `http://127.0.0.1:5010/predict`, expected output key `mic_score`.
- Hemo service: `http://127.0.0.1:5006/predict`, expected output key `scores`.

Output directory created by the command:

- `outputs/training/paper_mic_hemo_10000_off020_rand015_seed0/`

## Files Currently Pulled In Only Because of Inheritance/Top-Level Imports

These are not conceptually part of AMP-MOO, but they are currently imported
when `amp_moo.py` imports `SEHMOOTask` and `StandardOnlineTrainer`:

- `src/gflownet/tasks/seh_frag_moo.py`
- `src/gflownet/tasks/seh_frag.py`
- `src/gflownet/envs/graph_building_env.py`
- `src/gflownet/envs/frag_mol_env.py`
- `src/gflownet/models/graph_transformer.py`
- `src/gflownet/models/bengio2021flow.py`
- `src/gflownet/algo/trajectory_balance.py`
- `src/gflownet/algo/flow_matching.py`
- `src/gflownet/algo/advantage_actor_critic.py`
- `src/gflownet/algo/soft_q_learning.py`
- `src/gflownet/algo/envelope_q_learning.py`
- `src/gflownet/algo/multiobjective_reinforce.py`
- `src/gflownet/algo/graph_sampling.py`
- `src/gflownet/utils/sascore.py`
- `src/gflownet/utils/transforms.py`
- `src/gflownet/utils/focus_model.py`

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
`src/gflownet/tasks/amp_conditioning.py`, using only:

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
`src/gflownet/amp_trainer.py`, using only:

- `gflownet.data.sampling_iterator.SamplingIterator`
- `gflownet.data.replay_buffer.MOOReplayBuffer`
- `gflownet.config.Config`
- `gflownet.utils.misc.create_logger`

That refactor would allow deleting almost all graph/molecule files from the
public AMP-MOO repository.

## Release Scope

Historical experiment scripts, generated results, copied analysis directories,
unused oracle implementations, and backup files are not part of this release.
The remaining original graph/molecule modules must not be deleted until either:

1. `amp_moo.py` no longer imports `SEHMOOTask`, `StandardOnlineTrainer`, or
   `FlatRewards` from graph-coupled files; or
2. the top-level imports in those modules are made lazy and verified in a clean
   environment.
