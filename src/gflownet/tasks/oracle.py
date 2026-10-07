import torch
import requests
import os
from pathlib import Path
import sys


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validated_scores(values, expected_count: int, label: str, unit_interval: bool = False):
    if values is None:
        raise ValueError(f"{label} response is missing scores")
    scores = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if scores.numel() != expected_count:
        raise ValueError(
            f"{label} returned {scores.numel()} scores for {expected_count} sequences"
        )
    if not torch.isfinite(scores).all():
        raise ValueError(f"{label} returned NaN or infinite scores")
    if unit_interval and ((scores < 0).any() or (scores > 1).any()):
        raise ValueError(f"{label} returned scores outside [0, 1]")
    return scores


def _oracle_failure(label: str, exc: Exception, count: int, fallback: float):
    if _env_flag("ORACLE_ALLOW_FALLBACK", default=False):
        print(
            f"[WARNING] {label} failed; ORACLE_ALLOW_FALLBACK is enabled, "
            f"so {fallback} will be used for {count} sequences. Error: {exc}"
        )
        return torch.full((count,), fallback, dtype=torch.float32)
    raise RuntimeError(
        f"{label} failed. Refusing to train with fabricated rewards. "
        "Fix the reward service or explicitly set ORACLE_ALLOW_FALLBACK=1 "
        "for debugging only."
    ) from exc

# =====================================================================
# E. coli MIC Oracle
# =====================================================================
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_MIC_ORACLE_DIR = _REPO_ROOT / "services" / "mic"

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
        
        self.api_url = os.environ.get(
            "AMP_URL",
            f"http://127.0.0.1:{target_port}/predict",
        )
        
        print(f"[AMP Oracle] using service: {self.api_url}")

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
            response.raise_for_status()
            result = response.json()
            if result.get("status") != "success":
                raise ValueError(f"service status is not success: {result}")
            scores = _validated_scores(
                result.get("scores"), len(seqs), "AMP oracle", unit_interval=True
            )
        except Exception as exc:
            scores = _oracle_failure("AMP oracle", exc, len(seqs), fallback=0.0)

        return scores.to(self.device)

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
        self.api_url = os.environ.get(
            "HEMO_URL",
            f"http://127.0.0.1:{target_port}/predict",
        )
        print(f"[Hemo Oracle] using service: {self.api_url}")

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

            response.raise_for_status()
            result = response.json()
            if result.get("status") != "success":
                raise ValueError(f"service status is not success: {result}")
            scores = _validated_scores(
                result.get("scores"), len(seqs), "Hemo oracle", unit_interval=True
            )
        except Exception as exc:
            scores = _oracle_failure("Hemo oracle", exc, len(seqs), fallback=0.5)

        return scores.to(self.device)


# =====================================================================
# 🌟 3. HydrophobicMomentOracle (疏水矩裁判)
# =====================================================================
class HydrophobicMomentOracle:
    """
    成药机制裁判：利用 modlAMP 计算多肽的两亲性疏水矩 (Hydrophobic Moment)。
    """
    def __init__(self, device="cpu"):
        self.device = device
        print("[HMoment Oracle] modlAMP descriptor is ready")

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

            response.raise_for_status()
            result = response.json()
            if result.get("status") != "success":
                raise ValueError(f"service status is not success: {result}")
            scores = _validated_scores(
                result.get("scores"), len(seqs), "Tox oracle", unit_interval=True
            )
        except Exception as exc:
            scores = _oracle_failure("Tox oracle", exc, len(seqs), fallback=0.5)

        return scores.to(self.device)


# =====================================================================
# 4. EcoliMICOracleWrapper
# =====================================================================
class EcoliMICOracle:
    """
    E. coli-specific MIC oracle for OP-GFN.

    这个类包装 services/mic/ecoli_mic_oracle.py 里面的可选独立预测器，
    让 amp_moo.py 可以直接从 gflownet.tasks.oracle 导入它。

    predict(seqs) 返回的是 ecoli_score，越大越好。
    predict_log_mic(seqs) 返回的是 pred_log_mic，越低越好。
    """

    def __init__(self, device="cpu", model_path=None, batch_size=32):
        if _StandaloneEcoliMICOracle is None:
            raise ImportError(
                "Cannot import standalone EcoliMICOracle from services/mic/ecoli_mic_oracle.py. "
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
            f"[Ecoli MIC Oracle] loaded ESM2-Ridge predictor;"
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

        scores = _validated_scores(
            out[self.output_key], len(seqs), "BERT MIC oracle", unit_interval=True
        )
        return scores.numpy()

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

        scores = _validated_scores(
            out[self.output_key], len(seqs), "MBC MIC oracle", unit_interval=False
        )
        return scores.numpy()

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


def preflight_reward_oracles(
    objectives,
    mic_oracle_name="bert",
    sample_sequence="KIVRIFFKILKF",
):
    """Run one real prediction through every selected training reward oracle."""
    selected = list(objectives)
    oracles = {}
    if "amp" in selected:
        oracles["amp"] = AMPOracle(device="cpu")
    if "hemo" in selected:
        oracles["hemo_probability"] = HemoOracle(device="cpu")
    if "hmoment" in selected:
        oracles["hmoment"] = HydrophobicMomentOracle(device="cpu")
    if "mic" in selected:
        if mic_oracle_name == "bert":
            oracles["mic"] = BERTMICOracle(output_key="mic_score")
        elif mic_oracle_name == "mbc":
            oracles["mic"] = MBCMICOracle(output_key="pmic")
        else:
            raise ValueError(f"Unknown MIC oracle: {mic_oracle_name}")
    if "tox" in selected:
        oracles["tox_probability"] = ToxOracle(device="cpu")

    results = {}
    for name, oracle in oracles.items():
        score = oracle.predict([sample_sequence])
        score = _validated_scores(score, 1, name, unit_interval=True)
        results[name] = float(score.item())
    return results
