import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gflownet.utils.reproducibility import (
    ReproducibilityManifestError,
    build_amp_reproducibility_manifest,
    validate_amp_sampling_manifest,
)


AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")


def namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def make_context(motifs=None, min_length=5):
    motifs = list(motifs or ["KR", "GLR"])
    return namespace(
        vocab=AA_VOCAB,
        motifs=motifs,
        motif_action_start=20,
        motif_action_end=20 + len(motifs),
        stop_action_idx=20 + len(motifs),
        num_actions=21 + len(motifs),
        max_length=50,
        min_length=min_length,
        max_motif_actions=2,
        pad_idx=21,
    )


def make_model(select_gate_strength=0.75):
    attention = namespace(num_heads=8)
    layer = namespace(self_attn=attention)
    return namespace(
        embedding=namespace(embedding_dim=256),
        transformer=namespace(layers=[layer, layer, layer, layer]),
        use_motif_gate=True,
        motif_use_gate_strength=1.0,
        motif_select_gate_strength=select_gate_strength,
    )


class AMPReproducibilityManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.motif_path = root / "motifs.txt"
        self.offline_path = root / "offline.fasta"
        self.cache_path = root / "offline_mic_hemo_bert_rewards.pt"
        self.motif_path.write_text("KR\nGLR\n", encoding="utf-8")
        self.offline_path.write_text(">p1\nKRGLR\n", encoding="utf-8")
        self.cache_path.write_bytes(b"reward-cache")

        self.cfg = namespace(
            seed=7,
            model=namespace(
                num_emb=256,
                num_layers=4,
                graph_transformer=namespace(num_heads=8),
            ),
            replay=namespace(use=True, capacity=100000, warmup=1000, hindsight_ratio=0.0),
            algo=namespace(
                offline_ratio=0.2,
                train_random_action_prob=0.15,
                global_batch_size=64,
                sampling_tau=0.95,
            ),
        )
        self.args = namespace(
            motif_vocab_path=str(self.motif_path),
            reward_cache_tag="bert",
        )
        self.task = namespace(
            objectives=["mic", "hemo"],
            mic_oracle_name="bert",
            num_cond_dim=64,
        )
        self.ctx = make_context()
        self.model = make_model()
        self.algo = namespace(
            offline_stop_loss_weight=0.5,
            online_stop_len=20,
            online_stop_loss_weight=0.3,
            motif_usage_loss_weight=0.1,
            motif_diversity_loss_weight=0.05,
            pareto_len_limit=30,
            replay_len_limit=30,
        )
        self.manifest = build_amp_reproducibility_manifest(
            cfg=self.cfg,
            cmd_args=self.args,
            task=self.task,
            ctx=self.ctx,
            model=self.model,
            algo=self.algo,
            offline_data_path=self.offline_path,
            reward_cache_path=self.cache_path,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, **overrides):
        return validate_amp_sampling_manifest(
            overrides.get("manifest", self.manifest),
            motif_vocab_path=overrides.get("motif_vocab_path", self.motif_path),
            task=overrides.get("task", self.task),
            ctx=overrides.get("ctx", self.ctx),
            model=overrides.get("model", self.model),
        )

    def test_exact_reconstruction_passes(self):
        observed_hash = self.validate()
        self.assertEqual(observed_hash, self.manifest["manifest_sha256"])
        self.assertEqual(self.manifest["action_space"]["mapping"][20]["token"], "KR")
        self.assertIsNotNone(self.manifest["offline_data"]["sha256"])
        self.assertIsNotNone(self.manifest["reward_cache"]["sha256"])

    def test_changed_motif_file_is_rejected_even_with_same_count(self):
        changed = Path(self.tmp.name) / "changed.txt"
        changed.write_text("GLR\nKR\n", encoding="utf-8")
        with self.assertRaisesRegex(ReproducibilityManifestError, "motif_vocab.sha256"):
            self.validate(motif_vocab_path=changed, ctx=make_context(["GLR", "KR"]))

    def test_changed_action_order_is_rejected_even_when_file_hash_matches(self):
        with self.assertRaisesRegex(ReproducibilityManifestError, "effective_motifs"):
            self.validate(ctx=make_context(["GLR", "KR"]))

    def test_changed_gate_strength_is_rejected(self):
        changed_model = make_model(select_gate_strength=1.0)
        with self.assertRaisesRegex(ReproducibilityManifestError, "motif_select_gate_strength"):
            self.validate(model=changed_model)

    def test_changed_min_length_is_rejected(self):
        with self.assertRaisesRegex(ReproducibilityManifestError, "environment.min_length"):
            self.validate(ctx=make_context(min_length=6))

    def test_manifest_tampering_is_rejected(self):
        changed_manifest = copy.deepcopy(self.manifest)
        changed_manifest["environment"]["stop_action_idx"] += 1
        with self.assertRaisesRegex(ReproducibilityManifestError, "content hash mismatch"):
            self.validate(manifest=changed_manifest)


if __name__ == "__main__":
    unittest.main()
