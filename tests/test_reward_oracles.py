import json
import os
import unittest
from unittest.mock import Mock, patch

import torch

from gflownet.tasks.oracle import BERTMICOracle, HemoOracle, ToxOracle


class _URLResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RewardOracleContractTest(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_bert_mic_contract(self, urlopen):
        urlopen.return_value = _URLResponse({"mic_score": [0.25, 0.75]})
        oracle = BERTMICOracle(api_url="http://reward.test/predict")

        scores = oracle.predict(["AAAAA", "CCCCC"])

        self.assertTrue(torch.equal(scores, torch.tensor([0.25, 0.75])))

    @patch("urllib.request.urlopen")
    def test_bert_mic_rejects_wrong_score_count(self, urlopen):
        urlopen.return_value = _URLResponse({"mic_score": [0.25]})
        oracle = BERTMICOracle(api_url="http://reward.test/predict")

        with self.assertRaisesRegex(ValueError, "1 scores for 2 sequences"):
            oracle.predict(["AAAAA", "CCCCC"])

    @patch("gflownet.tasks.oracle.requests.post")
    def test_hemo_url_override_and_contract(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "success", "scores": [0.2]}
        post.return_value = response

        with patch.dict(os.environ, {"HEMO_URL": "http://reward.test/hemo"}):
            oracle = HemoOracle()
            scores = oracle.predict(["AAAAA"])

        self.assertEqual(oracle.api_url, "http://reward.test/hemo")
        self.assertAlmostEqual(float(scores.item()), 0.2, places=6)

    @patch("gflownet.tasks.oracle.requests.post")
    def test_hemo_failure_is_fatal_by_default(self, post):
        post.side_effect = OSError("service unavailable")
        with patch.dict(os.environ, {"ORACLE_ALLOW_FALLBACK": "0"}):
            with self.assertRaisesRegex(RuntimeError, "Refusing to train"):
                HemoOracle().predict(["AAAAA"])

    @patch("gflownet.tasks.oracle.requests.post")
    def test_tox_url_override_and_contract(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": "success", "scores": [0.3]}
        post.return_value = response

        with patch.dict(os.environ, {"TOX_URL": "http://reward.test/tox"}):
            oracle = ToxOracle()
            scores = oracle.predict(["AAAAA"])

        self.assertEqual(oracle.api_url, "http://reward.test/tox")
        self.assertAlmostEqual(float(scores.item()), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
