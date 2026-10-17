import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import draccus
import numpy as np
import pandas as pd

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.grbc.config import GraphConfig
from lerobot.policies.grbc.evaluate_checkpoint import checkpoint_rename_map
from lerobot.policies.grbc.evaluate_holdout import evaluate
from lerobot.policies.grbc.visualize_path import trace_path


class EvaluationTest(unittest.TestCase):
    def test_checkpoint_camera_map_restored_and_overridable(self):
        with tempfile.TemporaryDirectory() as directory:
            rename = {"observation.images.cam_0": "observation.images.camera1"}
            (Path(directory) / "policy_preprocessor.json").write_text(
                json.dumps(
                    {
                        "steps": [
                            {
                                "registry_name": "rename_observations_processor",
                                "config": {"rename_map": rename},
                            }
                        ]
                    }
                )
            )
            self.assertEqual(checkpoint_rename_map(directory, {}), rename)
            custom = {"observation.images.custom": "observation.images.camera1"}
            self.assertEqual(checkpoint_rename_map(directory, custom), custom)

    def test_checkpoint_without_processor_keeps_map(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(checkpoint_rename_map(directory, {}), {})

    def test_trace_success_sink(self):
        path, reached = trace_path(np.array([1, 2, 3]), 0)
        self.assertEqual(path, [0, 1, 2])
        self.assertTrue(reached)

    def test_trace_unreachable(self):
        path, reached = trace_path(np.array([-1, 2, 3]), 0)
        self.assertEqual(path, [0])
        self.assertFalse(reached)

    def test_trace_cycle_rejected(self):
        with self.assertRaises(ValueError):
            trace_path(np.array([1, 0]), 0)

    def fixture(self):
        nodes = pd.DataFrame(
            {
                "node_id": range(6),
                "index": range(6),
                "episode_index": [0, 0, 0, 1, 1, 1],
                "frame_index": [0, 15, 30, 0, 15, 30],
                "episode_length": 31,
                "next_reward": [0, 0, 1, 0, 0, 1],
                "next.reward": [0, 0, 1, 0, 0, 1],
                "episode_success": True,
                "gripper_closed": False,
            }
        )
        features = np.random.default_rng(9).normal(size=(6, 12)).astype(np.float32)
        return nodes, features

    def test_holdout_labels_do_not_create_goal_edges(self):
        nodes, features = self.fixture()
        values, _, _ = evaluate(nodes, features, GraphConfig(adaptive_edge_alpha=1e-8), [0], [1])
        np.testing.assert_array_equal(values.progress, np.zeros(3))
        nodes.loc[nodes.episode_index == 1, "next_reward"] = 0.0
        nodes.loc[nodes.episode_index == 1, "episode_success"] = False
        other, _, _ = evaluate(nodes, features, GraphConfig(adaptive_edge_alpha=1e-8), [0], [1])
        np.testing.assert_array_equal(values.progress, other.progress)

    def test_overlapping_split_rejected(self):
        nodes, features = self.fixture()
        with self.assertRaises(ValueError):
            evaluate(nodes, features, GraphConfig(), [0], [0])

    def test_query_reward_only_sample_removed(self):
        nodes, features = self.fixture()
        nodes.loc[4, "frame_index"] = 7
        values, _, _ = evaluate(nodes, features, GraphConfig(), [0], [1])
        self.assertEqual(values.frame_index.tolist(), [0, 30])

    def test_actual_shell_arguments_parse_for_all_backbones(self):
        root = Path(__file__).resolve().parents[2]
        env = {
            **os.environ,
            "DATASET_ROOT": "/example/task",
            "OUTPUT_DIR": "/example/train",
            "WEIGHTS_PATH": "/example/transition_weights.parquet",
        }
        for backbone in ("smolvla", "xvla", "pi05"):
            command = subprocess.check_output(
                ["bash", str(root / f"scripts/train_{backbone}.sh"), "--dry-run"], env=env, text=True
            )
            parts = shlex.split(command)
            args = parts[parts.index("lerobot.scripts.lerobot_train") + 1 :]
            # Avoid downloading a pretrained config while validating every launcher flag.
            args = [arg for arg in args if not arg.startswith("--policy.path=")]
            args = [f"--policy.type={backbone}", *args]
            config = draccus.parse(TrainPipelineConfig, args=args)
            self.assertEqual(config.policy.type, backbone)
            self.assertEqual(config.policy.n_action_steps, 30)
            self.assertEqual(config.policy.optimizer_betas, (0.9, 0.95))
            self.assertTrue(config.use_grbc)
            if backbone == "smolvla":
                self.assertFalse(config.policy.load_vlm_weights)
                self.assertEqual(
                    config.rename_map,
                    {f"observation.images.cam_{i}": f"observation.images.camera{i + 1}" for i in range(3)},
                )

    def test_short_run_warmup_override(self):
        root = Path(__file__).resolve().parents[2]
        env = {
            **os.environ,
            "DATASET_ROOT": "/example/task",
            "OUTPUT_DIR": "/example/train",
            "WEIGHTS_PATH": "/example/weights.parquet",
            "STEPS": "5",
            "WARMUP_STEPS": "1",
        }
        command = subprocess.check_output(
            ["bash", str(root / "scripts/train_smolvla.sh"), "--dry-run"], env=env, text=True
        )
        self.assertIn("--policy.scheduler_warmup_steps=1", shlex.split(command))


if __name__ == "__main__":
    unittest.main()
