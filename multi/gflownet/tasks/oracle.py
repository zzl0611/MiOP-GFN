import torch
import requests
import os
import tempfile
import subprocess
import pandas as pd
from pathlib import Path
import sys

# =====================================================================
# E. coli MIC Oracle
# =====================================================================
_THIS_FILE = Path(__file__).resolve()
_MULTI_ROOT = _THIS_FILE.parents[2]
_MIC_ORACLE_DIR = _MULTI_ROOT / "MIC_oracle"

if str(_MIC_ORACLE_DIR) not in sys.path:
    sys.path.append(str(_MIC_ORACLE_DIR))

try:
    from ecoli_mic_oracle import EcoliMICOracle as _StandaloneEcoliMICOracle
except Exception as e:
    _StandaloneEcoliMICOracle = None
    _ECOLI_MIC_IMPORT_ERROR = e

# 尝试导入多肽描述符计算工具包
try:
    from modlamp.descriptors import PeptideDescriptor
except ImportError:
    pass


# =====================================================================
# 🌟 1. AMPOracle (抗菌活性裁判)
# =====================================================================
class AMPOracle:
    def __init__(self, device="cpu"):
        self.device = device
        # 🌟 动态从环境变量读取，默认强制设为 5004，与高级版 5000 物理隔离！
        target_port = os.environ.get("AMP_PORT", "5007")
        
        self.api_url = f"http://127.0.0.1:{target_port}/predict"
        
        print(f"🚀 [AMP Oracle] 已切换为本地微服务极速模式！目标端口: {target_port}")

    @torch.no_grad()
    def predict(self, seqs: list[str]) -> torch.Tensor:
        if not seqs:
            return torch.tensor([], device=self.device)

        try:
            response = requests.post(
                self.api_url,
                json={"sequences": seqs},
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    scores = result.get("scores")
                else:
                    scores = [0.0] * len(seqs)
            else:
                scores = [0.0] * len(seqs)

        except requests.exceptions.RequestException:
            print(
                f"⚠️ 无法连接到 AMPlify 服务器 (端口 {self.api_url})！"
                "是不是忘记在终端运行 amplify_server.py 了？"
            )
            scores = [0.0] * len(seqs)

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

# =====================================================================
# 🌟 2. HemoOracle (溶血性裁判)
# =====================================================================
class HemoOracle:
    """
    调用本地 HemoPI2 微服务，返回 hemolytic probability。
    注意：这里返回的是溶血概率，越大越可能溶血；
    amp_moo.py 里会用 1.0 - hemo_probs 转成安全性分数。
    """
    def __init__(self, device="cpu"):
        self.device = device
        target_port = os.environ.get("HEMO_PORT", "5006")
        self.api_url = f"http://127.0.0.1:{target_port}/predict"
        print(f"🚀 [Hemo Oracle] 已切换为本地微服务模式！目标端口: {target_port}")

    @torch.no_grad()
    def predict(self, seqs: list[str]) -> torch.Tensor:
        if not seqs:
            return torch.tensor([], device=self.device)

        try:
            response = requests.post(
                self.api_url,
                json={"sequences": seqs},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    scores = result.get("scores", [0.5] * len(seqs))
                else:
                    scores = [0.5] * len(seqs)
            else:
                scores = [0.5] * len(seqs)

        except requests.exceptions.RequestException:
            print(f"⚠️ 无法连接到 HemoPI2 服务器 ({self.api_url})！是不是忘记启动 hemo_server.py 了？")
            scores = [0.5] * len(seqs)

        return torch.tensor(scores, dtype=torch.float32, device=self.device)


# =====================================================================
# 🌟 3. HydrophobicMomentOracle (疏水矩裁判)
# =====================================================================
class HydrophobicMomentOracle:
    """
    成药机制裁判：利用 modlAMP 计算多肽的两亲性疏水矩 (Hydrophobic Moment)。
    """
    def __init__(self, device="cpu"):
        self.device = device
        print("✅ [HMoment Oracle] modlAMP 物理化学计算引擎准备就绪！")

    def predict(self, seqs: list[str]) -> torch.Tensor:
        scores = []
        for seq in seqs:
            if len(seq) < 5:
                scores.append(0.0)
                continue
            try:
                desc = PeptideDescriptor(seq, 'eisenberg')
                desc.calculate_moment(angle=100)
                moment_value = float(desc.descriptor[0][0])
                scores.append(moment_value)
            except Exception:
                scores.append(0.0)
                
        return torch.tensor(scores, dtype=torch.float32, device=self.device)


# =====================================================================
# 🛡️ 备用武器库：ToxOracle (毒性裁判)
# =====================================================================
class ToxOracle:
    def __init__(self, device="cpu"):
        self.device = device
        target_port = os.environ.get("TOX_PORT", "5008")
        self.api_url = os.environ.get(
            "TOX_URL",
            f"http://127.0.0.1:{target_port}/predict",
        )
        print(f"[Tox Oracle] using local service: {self.api_url}")

    @torch.no_grad()
    def predict(self, seqs: list[str]) -> torch.Tensor:
        if not seqs:
            return torch.tensor([], device=self.device)

        try:
            response = requests.post(
                self.api_url,
                json={"sequences": seqs},
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    scores = result.get("scores", [0.5] * len(seqs))
                else:
                    scores = [0.5] * len(seqs)
            else:
                scores = [0.5] * len(seqs)

        except requests.exceptions.RequestException as e:
            print(f"Cannot connect to ToxPred3 service {self.api_url}: {e}")
            scores = [0.5] * len(seqs)

        return torch.tensor(scores, dtype=torch.float32, device=self.device)


# =====================================================================
# 4. EcoliMICOracleWrapper
# =====================================================================
class EcoliMICOracle:
    """
    E. coli-specific MIC oracle for OP-GFN.

    这个类包装 MIC_oracle/ecoli_mic_oracle.py 里面的独立预测器，
    让 amp_moo.py 可以直接从 gflownet.tasks.oracle 导入它。

    predict(seqs) 返回的是 ecoli_score，越大越好。
    predict_log_mic(seqs) 返回的是 pred_log_mic，越低越好。
    """

    def __init__(self, device="cpu", model_path=None, batch_size=32):
        if _StandaloneEcoliMICOracle is None:
            raise ImportError(
                "Cannot import standalone EcoliMICOracle from MIC_oracle/ecoli_mic_oracle.py. "
                f"Original error: {_ECOLI_MIC_IMPORT_ERROR}"
            )

        if model_path is None:
            model_path = (
                _MIC_ORACLE_DIR
                / "ecoli_mic_ridge_esm2_t12"
                / "ecoli_mic_ridge_esm2.joblib"
            )

        self.device = device
        self.model_path = str(model_path)
        self.batch_size = batch_size

        self.oracle = _StandaloneEcoliMICOracle(
            model_path=self.model_path,
            device=self.device,
            batch_size=self.batch_size,
        )

        print(
            f"✅ [Ecoli MIC Oracle] 已加载 ESM2-Ridge E. coli MIC 预测器！"
            f" model_path={self.model_path}"
        )

    @torch.no_grad()
    def predict_log_mic(self, seqs: list[str]) -> torch.Tensor:
        if not seqs:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        pred_log_mic = self.oracle.predict_log_mic(seqs)
        return pred_log_mic.to(self.device)

    @torch.no_grad()
    def predict(self, seqs: list[str]) -> torch.Tensor:
        """
        返回 OP-GFN 用的 E. coli MIC reward score。
        注意：score 越大越好，代表预测 MIC 越低。
        """
        if not seqs:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        score = self.oracle.predict(seqs)
        return score.to(self.device)

class BERTMICOracle:
    """
    HTTP wrapper for BERT-AmPEP60-style MIC oracle.

    Returns mic_score in [0, 1].
    Higher mic_score means lower predicted MIC and stronger antibacterial activity.
    """

    def __init__(self, api_url=None, output_key="mic_score", timeout=120):
        import os
        self.api_url = api_url or os.environ.get(
            "MIC_BERT_URL",
            f"http://127.0.0.1:{os.environ.get('MIC_BERT_PORT', '5010')}/predict"
        )
        self.output_key = output_key
        self.timeout = timeout

    def __call__(self, seqs):
        return self.score(seqs)

    def score(self, seqs):
        import json
        import urllib.request
        import numpy as np

        if isinstance(seqs, str):
            seqs = [seqs]

        payload = {"seqs": list(seqs)}
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode())

        if self.output_key not in out:
            raise KeyError(
                f"BERT MIC oracle response has no key '{self.output_key}'. "
                f"Available keys: {list(out.keys())}"
            )

        return np.asarray(out[self.output_key], dtype=np.float32)

    @torch.no_grad()
    def predict(self, seqs: list[str]) -> torch.Tensor:
        if not seqs:
            return torch.tensor([], dtype=torch.float32)

        scores = self.score(seqs)
        scores = torch.tensor(scores, dtype=torch.float32)

        # Bert-MIC 返回的是 OP-GFN 用的 mic_score，越大越好。
        # 这里做一次保护，避免服务端偶尔返回略超出 [0, 1] 的值。
        return torch.clamp(scores, min=0.0, max=1.0)

    def predict_raw(self, seqs):
        import json
        import urllib.request

        if isinstance(seqs, str):
            seqs = [seqs]

        payload = {"seqs": list(seqs)}
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode())

        return out


# Add torch-style predict() interface for BERTMICOracle.
# amp_moo.py expects oracle.predict(seqs).cpu().
# def _bert_mic_oracle_predict(self, seqs):
#     import torch
#     scores = self.score(seqs)
#     return torch.tensor(scores, dtype=torch.float32)


# BERTMICOracle.predict = _bert_mic_oracle_predict

class MBCMICOracle:
    """
    HTTP wrapper for MBC-Attention MIC oracle.

    Server returns raw pMIC values.
    predict() maps pMIC through sigmoid to [0, 1].
    Higher score means stronger predicted antibacterial activity.
    """

    def __init__(self, api_url=None, output_key="pmic", timeout=120):
        import os
        self.api_url = api_url or os.environ.get(
            "MBC_MIC_URL",
            f"http://127.0.0.1:{os.environ.get('MBC_MIC_PORT', '5011')}/predict"
        )
        self.output_key = output_key
        self.timeout = timeout

    def __call__(self, seqs):
        return self.score(seqs)

    def score(self, seqs):
        import json
        import urllib.request
        import numpy as np

        if isinstance(seqs, str):
            seqs = [seqs]

        payload = {"seqs": list(seqs)}
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode())

        if self.output_key not in out:
            raise KeyError(
                f"MBC MIC oracle response has no key '{self.output_key}'. "
                f"Available keys: {list(out.keys())}"
            )

        return np.asarray(out[self.output_key], dtype=np.float32)

    @torch.no_grad()
    def predict(self, seqs: list[str]) -> torch.Tensor:
        if not seqs:
            return torch.tensor([], dtype=torch.float32)

        pmic = self.score(seqs)
        pmic = torch.tensor(pmic, dtype=torch.float32)

        return torch.sigmoid(pmic)

    def predict_raw(self, seqs):
        import json
        import urllib.request

        if isinstance(seqs, str):
            seqs = [seqs]

        payload = {"seqs": list(seqs)}
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())