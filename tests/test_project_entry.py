import ast
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_PATH = REPO_ROOT / "run.py"

spec = importlib.util.spec_from_file_location("miop_entry", ENTRY_PATH)
entry = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(entry)


class ProjectEntryTest(unittest.TestCase):
    def test_mic_predictor_resources_are_inside_repository(self):
        expected = (
            entry.PREDICTOR_ROOT / "bert_mic/server.py",
            entry.PREDICTOR_ROOT / "bert_mic/regressor.py",
            entry.PREDICTOR_ROOT / "bert_mic/artifacts/backbone/config.json",
            entry.PREDICTOR_ROOT / "bert_mic/artifacts/backbone/pytorch_model.bin",
            entry.PREDICTOR_ROOT / "bert_mic/artifacts/backbone/vocab.txt",
            entry.PREDICTOR_ROOT / "bert_mic/artifacts/best_model_state_dict.pkl",
            entry.PREDICTOR_ROOT / "bert_mic/artifacts/score_calibration.json",
        )
        self.assertEqual(entry.MIC_SERVER, expected[0])
        for path in expected:
            self.assertTrue(path.is_file(), path)

    def test_presets_reference_existing_training_resources(self):
        for name in ("mic-hemo", "mic-hemo-tox"):
            preset = entry.load_preset(name)
            args = preset["train_args"]
            for option in ("--offline_data", "--motif_vocab_path"):
                value = entry.last_option_value(args, option)
                self.assertIsNotNone(value)
                self.assertTrue((entry.REPO_ROOT / value).is_file(), (name, option, value))

    def test_presets_are_valid_json(self):
        for path in entry.CONFIG_ROOT.glob("*.json"):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["name"], path.stem)
            self.assertTrue(payload["objectives"])
            self.assertTrue(payload["train_args"])

    def test_mic_hemo_preset_matches_paper_method(self):
        args = entry.load_preset("mic-hemo")["train_args"]
        expected_values = {
            "--objectives": "mic",
            "--mic_oracle": "bert",
            "--type": "ordering",
            "--seed": "0",
            "--num_training_steps": "10000",
            "--offline_ratio": "0.20",
            "--offline_reward_batch_size": "64",
            "--global_batch_size": "64",
            "--reward_cache_tag": "bert_ec_final_mic_priority_motif67",
            "--max_motif_actions": "2",
            "--motif_use_gate_strength": "1.0",
            "--motif_select_gate_strength": "1.0",
            "--motif_usage_loss_weight": "0.0",
            "--pareto_len_limit": "30",
            "--replay_len_limit": "30",
            "--train_random_action_prob": "0.15",
        }
        for option, expected in expected_values.items():
            self.assertEqual(entry.last_option_value(args, option), expected, option)
        for flag in (
            "--compute_hvi",
            "--compute_igd",
            "--compute_pc_entropy",
            "--replay",
            "--use_motif_gate",
        ):
            self.assertIn(flag, args)
        objective_index = args.index("--objectives")
        self.assertEqual(args[objective_index + 1 : objective_index + 3], ["mic", "hemo"])

    def test_amp_moo_code_defaults_match_paper_method(self):
        source_path = entry.SRC_ROOT / "gflownet/tasks/amp_moo.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        defaults = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "add_argument":
                continue
            try:
                option = ast.literal_eval(node.args[0])
            except (ValueError, TypeError):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[option] = ast.literal_eval(keyword.value)

        expected = {
            "--log_dir": "./outputs/training/paper_mic_hemo_10000_off020_rand015_seed0",
            "--objectives": ["mic", "hemo"],
            "--replay": True,
            "--seed": 0,
            "--type": "ordering",
            "--compute_hvi": True,
            "--compute_igd": True,
            "--compute_pc_entropy": True,
            "--offline_data": "./datasets/offline/bert_ec_final_offline_mic_priority_top2000.fasta",
            "--offline_ratio": 0.20,
            "--offline_reward_batch_size": 64,
            "--global_batch_size": 64,
            "--mic_oracle": "bert",
            "--motif_vocab_path": "./datasets/motifs/bpe_motif_vocab_bert_ec_final_mic_hemo_notox_effect005.txt",
            "--max_motif_actions": 2,
            "--train_random_action_prob": 0.15,
            "--num_training_steps": 10000,
            "--use_motif_gate": True,
            "--motif_use_gate_strength": 1.0,
            "--motif_select_gate_strength": 1.0,
            "--motif_usage_loss_weight": 0.0,
            "--pareto_len_limit": 30,
            "--reward_cache_tag": "bert_ec_final_mic_priority_motif67",
            "--replay_len_limit": 30,
        }
        for option, expected_value in expected.items():
            self.assertEqual(defaults.get(option), expected_value, option)
        removed = {
            "--offline_stop_loss_weight",
            "--online_stop_len",
            "--online_stop_loss_weight",
            "--motif_diversity_loss_weight",
        }
        self.assertTrue(removed.isdisjoint(defaults))

    def test_train_strips_wrapper_separator(self):
        args = entry.build_parser().parse_args(
            ["train", "--preset", "mic-hemo", "--dry-run", "--", "--seed", "42"]
        )
        with patch.object(entry, "run_command", return_value=0) as run_command:
            self.assertEqual(entry.command_train(args), 0)
        command = run_command.call_args.args[0]
        self.assertNotIn("--", command)
        self.assertEqual(entry.last_option_value(command, "--seed"), "42")

    def test_user_paths_are_resolved_from_repository_root(self):
        path = entry.resolve_user_path("outputs/example.csv")
        self.assertEqual(path, (REPO_ROOT / "outputs/example.csv").resolve())


if __name__ == "__main__":
    unittest.main()
