import os
import re
import tempfile
import subprocess
from typing import Dict, List

import pandas as pd
from flask import Flask, request, jsonify


VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
TOXINPRED3_BIN = os.environ.get(
    "TOXINPRED3_BIN",
    "/home/lzz/anaconda3/envs/toxinpred3_env/bin/toxinpred3",
)

app = Flask(__name__)
CACHE: Dict[str, float] = {}


def clean_sequence(seq: str) -> str:
    seq = str(seq).upper()
    seq = re.sub(f"[^{VALID_AA}]", "", seq)
    if len(seq) < 5:
        seq = seq + "A" * (5 - len(seq))
    return seq


def run_toxinpred3(seqs: List[str]) -> List[float]:
    """
    调用 ToxinPred3。

    ToxinPred3 对单条 FASTA 输入存在 NumPy 维度错误。
    因此，当实际未缓存序列只有一条时，在服务内部临时加入
    一条 dummy 序列，并在预测结束后仅返回原始序列结果。
    """
    if not seqs:
        return []

    original_count = len(seqs)
    service_seqs = list(seqs)

    if original_count == 1:
        dummy_sequence = "ACDEFGHIKLMNPQRSTVWY"

        if service_seqs[0] == dummy_sequence:
            dummy_sequence = "KIVRIFFKILKF"

        service_seqs.append(dummy_sequence)

        print(
            "[ToxPred3] Singleton uncached request detected; "
            "one dummy sequence was appended.",
            flush=True,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        fasta_path = os.path.join(
            temp_dir,
            "input.fa",
        )

        out_csv_path = os.path.join(
            temp_dir,
            "out.csv",
        )

        with open(
            fasta_path,
            "w",
            encoding="utf-8",
        ) as handle:
            for index, sequence in enumerate(service_seqs):
                handle.write(
                    f">seq_{index}\n{sequence}\n"
                )

        cmd = [
            TOXINPRED3_BIN,
            "-i",
            fasta_path,
            "-o",
            out_csv_path,
            "-m",
            "2",
            "-d",
            "2",
        ]

        completed = subprocess.run(
            cmd,
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "ToxinPred3 command failed.\n"
                f"returncode={completed.returncode}\n"
                f"stdout={completed.stdout[-2000:]}\n"
                f"stderr={completed.stderr[-2000:]}"
            )

        if not os.path.exists(out_csv_path):
            raise FileNotFoundError(
                f"ToxinPred3未生成输出文件：{out_csv_path}"
            )

        with open(
            out_csv_path,
            "r",
            encoding="utf-8",
        ) as handle:
            lines = handle.readlines()

        start_idx = None

        for index, line in enumerate(lines):
            if line.startswith("Subject,ML Score"):
                start_idx = index
                break

        if start_idx is None:
            raise RuntimeError(
                "ToxinPred3输出中未找到CSV表头。\n"
                + "".join(lines[:20])
            )

        df = pd.read_csv(
            out_csv_path,
            skiprows=start_idx,
        )

        required_columns = {
            "Subject",
            "ML Score",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise KeyError(
                "ToxinPred3输出缺少字段："
                f"{sorted(missing_columns)}；"
                f"实际字段={df.columns.tolist()}"
            )

        score_dict = {
            str(row["Subject"]): float(row["ML Score"])
            for _, row in df.iterrows()
        }

        expected_subjects = [
            f"seq_{index}"
            for index in range(len(service_seqs))
        ]

        missing_subjects = [
            subject
            for subject in expected_subjects
            if subject not in score_dict
        ]

        if missing_subjects:
            raise RuntimeError(
                "ToxinPred3输出缺少序列："
                f"{missing_subjects}"
            )

        service_scores = [
            score_dict[subject]
            for subject in expected_subjects
        ]

        return service_scores[:original_count]


def predict_batch(raw_seqs: List[str]) -> List[float]:
    cleaned = [clean_sequence(s) for s in raw_seqs]
    final_scores = [None] * len(cleaned)

    uncached = []
    uncached_indices = []

    for i, seq in enumerate(cleaned):
        if seq in CACHE:
            final_scores[i] = CACHE[seq]
        else:
            uncached.append(seq)
            uncached_indices.append(i)

    if uncached:
        uncached_scores = run_toxinpred3(uncached)
        for idx, seq, score in zip(uncached_indices, uncached, uncached_scores):
            CACHE[seq] = float(score)
            final_scores[idx] = float(score)

    return [float(x) for x in final_scores]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        seqs = data.get("sequences", data.get("seqs", []))

        scores = predict_batch(seqs)

        return jsonify({
            "status": "success",
            "scores": scores,
            "cache_size": len(CACHE),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "failed",
            "error": str(e),
        }), 500


if __name__ == "__main__":
    run_port = int(os.environ.get("TOX_PORT", 5008))
    print(f"ToxPred3 service: http://127.0.0.1:{run_port}/predict")
    print(f"TOXINPRED3_BIN={TOXINPRED3_BIN}")
    app.run(host="127.0.0.1", port=run_port, threaded=False)