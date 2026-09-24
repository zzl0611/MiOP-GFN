import os
import re
from typing import List, Dict

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, EsmForSequenceClassification


# ==========================================================
# 1. HemoPI2 本地模型路径
# ==========================================================
MODEL_DIR = "/home/lzz/anaconda3/envs/op_gfn_mo/lib/python3.9/site-packages/hemopi2/Model"

VALID_AA = "ACDEFGHIKLMNPQRSTVWY"

app = Flask(__name__)

print("=" * 60)
print("⏳ 正在启动 HemoPI2 ESM GPU 微服务...")
print(f"📂 HemoPI2 模型路径: {MODEL_DIR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 HemoPI2 推理设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = EsmForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.to(device)
model.eval()

# 简单缓存：同一条序列算过一次就不再重复算
CACHE: Dict[str, float] = {}

print("✅ HemoPI2 ESM 模型已常驻内存！")
print("=" * 60)


def clean_sequence(seq: str) -> str:
    """
    对齐 HemoPI2 官方逻辑：
    1. 只保留 20 种标准氨基酸
    2. 超过 40 aa 截断到前 40 aa
    3. 太短的序列补 A 到 5 aa，防止模型异常
    """
    seq = str(seq).upper()
    seq = re.sub(f"[^{VALID_AA}]", "", seq)

    if len(seq) < 5:
        seq = seq + "A" * (5 - len(seq))

    if len(seq) > 40:
        seq = seq[:40]

    return seq


@torch.no_grad()
def predict_batch(seqs: List[str], batch_size: int = 64) -> List[float]:
    """
    返回 hemolytic probability：
    越大表示越可能溶血；
    你的训练里会用 1 - prob 作为 Hemo safety。
    """
    cleaned = [clean_sequence(s) for s in seqs]

    final_scores = [None] * len(cleaned)

    uncached = []
    uncached_indices = []

    for i, seq in enumerate(cleaned):
        if seq in CACHE:
            final_scores[i] = CACHE[seq]
        else:
            uncached.append(seq)
            uncached_indices.append(i)

    if len(uncached) > 0:
        all_uncached_scores = []

        for start in range(0, len(uncached), batch_size):
            sub_seqs = uncached[start:start + batch_size]

            inputs = tokenizer(
                sub_seqs,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)[:, 1]
            probs = probs.detach().cpu().float().tolist()

            all_uncached_scores.extend(probs)

        for idx, score in zip(uncached_indices, all_uncached_scores):
            seq_key = cleaned[idx]
            CACHE[seq_key] = float(score)
            final_scores[idx] = float(score)

    return [float(x) for x in final_scores]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        seqs = data.get("sequences", [])

        if not seqs:
            return jsonify({
                "status": "success",
                "scores": []
            })

        batch_size = int(data.get("batch_size", os.environ.get("HEMO_BATCH_SIZE", 64)))

        scores = predict_batch(seqs, batch_size=batch_size)

        return jsonify({
            "status": "success",
            "scores": scores,
            "cache_size": len(CACHE)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    run_port = int(os.environ.get("HEMO_PORT", 5006))
    print(f"📡 HemoPI2 微服务监听地址: http://127.0.0.1:{run_port}/predict")
    app.run(host="127.0.0.1", port=run_port, threaded=False)