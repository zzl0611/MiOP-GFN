# Local reward services

AMP-MOO training uses reward predictors through HTTP. The predictors do not
need to live inside this repository, but every selected objective must have a
working service before online training starts.

## Service contract

| Objective | Environment variable | Default endpoint | Response used by training |
| --- | --- | --- | --- |
| BERT MIC | `MIC_BERT_URL` | `http://127.0.0.1:5010/predict` | `mic_score` |
| MBC MIC | `MBC_MIC_URL` | `http://127.0.0.1:5011/predict` | `pmic` |
| Hemolysis | `HEMO_URL` | `http://127.0.0.1:5006/predict` | `scores` |
| Toxicity | `TOX_URL` | `http://127.0.0.1:5008/predict` | `scores` |
| AMP | `AMP_URL` | `http://127.0.0.1:5007/predict` | `scores` |

The BERT/MBC requests use `{"seqs": [...]}`. Hemo, Tox, and AMP requests use
`{"sequences": [...]}`. Every selected service must return exactly one finite
score per input sequence. Probability-like scores must be in `[0, 1]`.

Training now performs one real prediction with each selected oracle before it
creates the trainer. A failed or malformed service stops the run instead of
silently replacing rewards with constants. `ORACLE_ALLOW_FALLBACK=1` restores
the old fallback behavior for debugging only and must not be used for a real
experiment.

## Option A: use already deployed services

This is the simplest setup when the three predictors already run elsewhere:

```powershell
$env:MIC_BERT_URL = 'http://SERVER:5010/predict'
$env:HEMO_URL = 'http://SERVER:5006/predict'
$env:TOX_URL = 'http://SERVER:5008/predict'
```

No predictor model files are then required on the MiOP-GFN machine. The remote
service must be reachable from the training process.

## Option B: run all three services locally

Use separate terminals. The services may use separate Python environments;
only the HTTP contracts above are shared with MiOP-GFN.

### 1. BERT EC MIC

The repository keeps the BERT MIC service and its backbone, fine-tuned
checkpoint, and calibration file together in `predictors/bert_mic/`.
From the MiOP-GFN repository root:

```powershell
$env:MIC_BERT_PORT = '5010'
python .\run.py serve mic
```

Optional overrides are `BERT_MODEL_PATH`, `BERT_TOKENIZER_PATH`,
`BERT_EC_CKPT`, and `BERT_EC_CALIB`.

### 2. HemoPI2

Point `HEMO_MODEL_DIR` at the local Hugging Face-style HemoPI2 `Model`
directory. If the `hemopi2` package is installed, the server also tries the
package's `Model` directory automatically.

```powershell
$env:HEMO_MODEL_DIR = 'D:\path\to\hemopi2\Model'
$env:HEMO_PORT = '5006'
python '.\MiOP-GFN\predictors\hemolysis\server.py'
```

### 3. ToxinPred3

Set the absolute executable path, or make `toxinpred3` available on `PATH`:

```powershell
$env:TOXINPRED3_BIN = 'D:\path\to\toxinpred3.exe'
$env:TOX_PORT = '5008'
python '.\MiOP-GFN\predictors\toxicity\server.py'
```

For services exposed to another machine, set the matching `*_HOST` variable
(`MIC_BERT_HOST`, `HEMO_HOST`, or `TOX_HOST`) to an appropriate bind address.
Do not expose an unauthenticated development server to an untrusted network.

## Start training

Activate the MiOP-GFN environment and run from the repository root. The preflight
is enabled by default:

```powershell
Set-Location '.\MiOP-GFN'
python .\run.py check --preset mic-hemo-tox
python .\run.py train `
  --preset mic-hemo-tox `
  --steps 10 `
  --log-dir .\outputs\training\local-smoke-mic-hemo-tox
```

After this smoke run succeeds, increase `--num_training_steps` and use the
experiment's full hyperparameter set. `--skip_oracle_preflight` exists only for
deliberate offline/debug work.
