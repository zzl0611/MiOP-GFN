import os
import json
from pathlib import Path
import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer

from regressor import REG


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_BACKBONE_DIR = ARTIFACT_DIR / "backbone"
PORT = int(os.environ.get("MIC_BERT_PORT", "5010"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = os.environ.get(
    "BERT_EC_CKPT",
    str(ARTIFACT_DIR / "best_model_state_dict.pkl"),
)

CALIB_PATH = os.environ.get(
    "BERT_EC_CALIB",
    str(ARTIFACT_DIR / "score_calibration.json"),
)

TOKENIZER_PATH = os.environ.get(
    "BERT_TOKENIZER_PATH",
    str(DEFAULT_BACKBONE_DIR),
)

MODEL_PATH = os.environ.get(
    "BERT_MODEL_PATH",
    str(DEFAULT_BACKBONE_DIR),
)

MAX_LEN = int(os.environ.get("BERT_EC_MAX_LEN", "77"))
BATCH_SIZE = int(os.environ.get("BERT_EC_BATCH_SIZE", "64"))

app = Flask(__name__)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clean_seq(seq):
    return "".join(str(seq).strip().upper().split())


def encode_batch(seqs):
    spaced = [" ".join(s) for s in seqs]
    encoded = tokenizer.batch_encode_plus(
        spaced,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].to(DEVICE),
        encoded["attention_mask"].to(DEVICE),
    )


def predict_pmic(seqs):
    preds = []
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE]
            input_ids, mask = encode_batch(batch)
            out, _ = model(input_ids, attention_mask=mask)
            preds.extend(out.detach().cpu().numpy().reshape(-1).tolist())
    return np.array(preds, dtype=float)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "checkpoint": CKPT_PATH,
        "calibration": CALIB,
        "tokenizer": TOKENIZER_PATH,
        "model": MODEL_PATH,
        "score_type": "calibrated_mic_score",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    seqs = data.get("seqs")
    if seqs is None:
        seq = data.get("seq")
        if seq is None:
            return jsonify({
                "status": "error",
                "message": "Missing field: seqs or seq",
            }), 400
        seqs = [seq]

    seqs = [clean_seq(s) for s in seqs]

    pred_pmic = predict_pmic(seqs)
    pred_mic_uM = np.power(10.0, -pred_pmic)

    mic_score = sigmoid((pred_pmic - PMIC_CENTER) / TEMP)
    mic_score = np.clip(mic_score, 0.0, 1.0)

    return jsonify({
        "status": "success",
        "score_type": "calibrated_mic_score",
        "seqs": seqs,
        "pred_pmic": pred_pmic.tolist(),
        "pred_mic_uM": pred_mic_uM.tolist(),
        "mic_score": mic_score.tolist(),
        "scores": mic_score.tolist(),
        "calibration": {
            "pmic_center": PMIC_CENTER,
            "temperature": TEMP,
            "formula": "mic_score = sigmoid((pred_pmic - pmic_center) / temperature)",
        },
    })


print("========== BERT EC Final MIC Server ==========")
print("repository:", REPO_ROOT)
print("checkpoint:", CKPT_PATH)
print("calibration:", CALIB_PATH)
print("tokenizer:", TOKENIZER_PATH)
print("model:", MODEL_PATH)
print("device:", DEVICE)
print("port:", PORT)
print("batch_size:", BATCH_SIZE)
print("max_len:", MAX_LEN)

if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")

if not os.path.exists(CALIB_PATH):
    raise FileNotFoundError(f"Missing calibration file: {CALIB_PATH}")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Missing tokenizer path: {TOKENIZER_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model path: {MODEL_PATH}")

with open(CALIB_PATH, "r") as f:
    CALIB = json.load(f)

PMIC_CENTER = float(CALIB["pmic_q50"])
TEMP = float(CALIB["temperature"])

tokenizer = BertTokenizer.from_pretrained(
    TOKENIZER_PATH,
    do_lower_case=False,
)

model = REG(bert_model_path=MODEL_PATH)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.to(DEVICE)
model.eval()

print("model loaded")
print("pmic_center:", PMIC_CENTER)
print("temperature:", TEMP)


if __name__ == "__main__":
    app.run(host=os.environ.get("MIC_BERT_HOST", "127.0.0.1"), port=PORT)
