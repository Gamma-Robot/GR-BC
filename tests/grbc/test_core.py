import ast
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from lerobot.policies.grbc.config import PRESETS, GraphConfig
from lerobot.policies.grbc.dataset import (
    history_rows,
)
from lerobot.policies.grbc.graph import (
    _camera_mutual_pairs,
    _split_camera_history_features,
    build_multi_view_adaptive_similarity_edges,
    build_temporal_edges,
    reverse_dijkstra,
)
from lerobot.policies.grbc.progress import dense_progress
from lerobot.policies.grbc.training import GRBCWeights, weighted_loss
from lerobot.policies.grbc.weights import compute_graph_transition_weights


def frame_table(progress):
    return pd.DataFrame({"episode_index": 0, "frame_index": np.arange(len(progress)), "progress": progress})


class WeightsTest(unittest.TestCase):
    def apply(self, p, **kwargs):
        return compute_graph_transition_weights(
            frame_table(p),
            chunk=3,
            n_action_step=3,
            progress_dip_tolerance=0.03,
            min_delta_progress=0.005,
            **kwargs,
        )

    def test_residual(self):
        result = self.apply([0, 0.02, 0.04, 0.1])
        self.assertAlmostEqual(result.graph_transition_weight.iloc[0], 1 + 0.5 * np.tanh(2), places=6)

    def test_dip_rejects_positive_endpoint(self):
        self.assertEqual(self.apply([0.3, 0.2, 0.4, 0.5]).graph_transition_weight.iloc[0], 0)

    def test_small_dip_passes(self):
        self.assertGreater(self.apply([0.3, 0.28, 0.4, 0.5]).graph_transition_weight.iloc[0], 1)

    def test_flat_and_negative_zero(self):
        self.assertEqual(self.apply([0.3, 0.3, 0.3, 0.3]).graph_transition_weight.iloc[0], 0)
        self.assertEqual(self.apply([0.3, 0.29, 0.29, 0.29]).graph_transition_weight.iloc[0], 0)

    def test_minimum_improvement_strict(self):
        self.assertEqual(self.apply([0, 0.001, 0.003, 0.005]).graph_transition_weight.iloc[0], 0)

    def test_last_frame_zero_and_tail_shortens(self):
        result = self.apply([0, 0.1, 0.2, 0.3])
        self.assertEqual(result.graph_transition_weight.iloc[-1], 0)
        self.assertEqual(result.judged_endpoint_offset.iloc[-2], 1)

    def test_judgment_not_prediction_horizon(self):
        p = [0, 0.1, 0.2, 0.3, 0.1, 0]
        result = compute_graph_transition_weights(
            frame_table(p), chunk=5, n_action_step=3, progress_dip_tolerance=0.03
        )
        self.assertGreater(result.graph_transition_weight.iloc[0], 1)

    def test_never_cross_episode(self):
        first = frame_table([0.1, 0.2])
        second = frame_table([0, 0])
        second.episode_index = 1
        result = compute_graph_transition_weights(pd.concat([first, second]), chunk=30)
        self.assertEqual(result.iloc[1].graph_transition_weight, 0)

    def test_invalid_horizon(self):
        with self.assertRaises(ValueError):
            compute_graph_transition_weights(frame_table([0, 1]), chunk=3, n_action_step=4)


class ProgressTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {"episode_index": 0, "frame_index": np.arange(5), "next.reward": [0, 0, 0, 0, 1]}
        )
        self.nodes = pd.DataFrame({"episode_index": 0, "frame_index": [0, 2, 4], "node_id": [0, 1, 2]})

    def test_finite_linear(self):
        table, rmax = dense_progress(self.data, self.nodes, np.array([4.0, 2.0, 0.0]), 1.0, "paper")
        np.testing.assert_allclose(table.progress, np.linspace(0, 1, 5))
        self.assertEqual(rmax, 4)

    def test_unreachable_pair_interior_zero(self):
        table, _ = dense_progress(self.data, self.nodes, np.array([np.inf, 2.0, 0.0]), 1.0, "paper")
        self.assertEqual(table.progress.iloc[1], 0)
        self.assertEqual(table.progress.iloc[3], 0.5)

    def test_finite_anchor_preserved_next_to_unreachable(self):
        table, _ = dense_progress(self.data, self.nodes, np.array([0.0, np.inf, 2.0]), 1.0, "paper")
        self.assertEqual(table.progress.iloc[0], 1)
        self.assertEqual(table.progress.iloc[1], 0)

    def test_clipping_order_differs(self):
        a, _ = dense_progress(self.data, self.nodes, np.array([100.0, 0.0, 0.0]), 0.5, "paper")
        b, _ = dense_progress(self.data, self.nodes, np.array([100.0, 0.0, 0.0]), 0.5, "legacy")
        self.assertNotEqual(a.progress.iloc[1], b.progress.iloc[1])

    def test_all_unreachable_rejected(self):
        with self.assertRaises(ValueError):
            dense_progress(self.data, self.nodes, np.full(3, np.inf))

    def test_single_frame(self):
        result, _ = dense_progress(self.data.iloc[:1], self.nodes.iloc[:1], np.array([0.0]))
        self.assertEqual(result.progress.iloc[0], 1)


class GraphTest(unittest.TestCase):
    def setUp(self):
        self.nodes = pd.DataFrame(
            {
                "node_id": np.arange(6),
                "episode_index": [0, 0, 0, 1, 1, 1],
                "frame_index": [0, 10, 20, 0, 10, 20],
            }
        )

    def test_history_padding_and_episode_boundary(self):
        mapping = history_rows(self.nodes, 2)
        np.testing.assert_array_equal(mapping[0], [0, 0, 0])
        np.testing.assert_array_equal(mapping[4], [4, 3, 3])

    def test_history_is_per_view_concat(self):
        features = np.arange(1, 37, dtype=np.float32).reshape(6, 6)
        result = _split_camera_history_features(features, history_rows(self.nodes, 2), 3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, (6, 6))
        np.testing.assert_allclose(np.linalg.norm(result[0], axis=1), 1, atol=1e-6)

    def test_tie_self_exclusion(self):
        pairs, _ = _camera_mutual_pairs(np.ones((4, 3)), 3)
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(i != j for i, j in pairs))

    def test_temporal_actual_frame_cost(self):
        edges = build_temporal_edges(self.nodes)
        self.assertEqual(edges, [(0, 1, 10.0), (1, 2, 10.0), (3, 4, 10.0), (4, 5, 10.0)])

    def test_reverse_search_direction_and_success(self):
        distance, nxt, _ = reverse_dijkstra(6, build_temporal_edges(self.nodes), [], {2})
        np.testing.assert_allclose(distance[:3], [20, 10, 0])
        self.assertTrue(np.isinf(distance[3:]).all())
        self.assertEqual(nxt[0], 1)

    def test_similarity_bidirectional_recovery(self):
        d, _, _ = reverse_dijkstra(6, build_temporal_edges(self.nodes), [(1, 4, 2.0, 0.2)], {2})
        self.assertEqual(d[4], 12)
        self.assertTrue(np.isinf(d[5]))

    def test_three_camera_intersection_and_gripper(self):
        features = np.random.default_rng(42).normal(size=(6, 12)).astype(np.float32)
        common = {
            "features": features,
            "nodes": self.nodes,
            "gripper_closed": np.zeros(6, dtype=bool),
            "mutual_k": 2,
            "stride": 10,
            "adaptive_alpha": 100,
            "adaptive_base_scale": 0.05,
            "cost_scale": 1,
        }
        intersection, stats = build_multi_view_adaptive_similarity_edges(**common, min_camera_hits=3)
        union, _ = build_multi_view_adaptive_similarity_edges(**common, min_camera_hits=1)
        self.assertTrue({e[:2] for e in intersection}.issubset({e[:2] for e in union}))
        self.assertEqual(stats["candidate_mknn_pairs"], stats["candidate_full_intersection_pairs"])
        common["gripper_closed"] = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
        edges, _ = build_multi_view_adaptive_similarity_edges(**common, min_camera_hits=1)
        self.assertTrue(all(common["gripper_closed"][i] == common["gripper_closed"][j] for i, j, *_ in edges))


class TrainingTest(unittest.TestCase):
    def test_raw_weight_mean(self):
        loss = torch.tensor([2.0, 4.0], requires_grad=True)
        value = weighted_loss(loss, torch.tensor([0.0, 1.5]))
        self.assertEqual(value.item(), 3)
        value.backward()
        torch.testing.assert_close(loss.grad, torch.tensor([0.0, 0.75]))

    def test_legacy_normalization(self):
        value = weighted_loss(torch.tensor([2.0, 4.0]), torch.tensor([0.0, 1.5]), reduction="self_normalized")
        self.assertEqual(value.item(), 4)

    def test_ddp_global_denominator(self):
        accelerator = SimpleNamespace(num_processes=2, reduce=lambda x, reduction: torch.tensor(6.0))
        value = weighted_loss(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 1.0]), accelerator, "mean")
        self.assertEqual(value.item(), 2)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            weighted_loss(torch.ones(3, 1), torch.ones(3))

    def test_provider_validates_indices_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.parquet"
            pd.DataFrame({"index": [10, 11], "graph_transition_weight": [0.0, 1.5]}).to_parquet(path)
            provider = GRBCWeights(path)
            weights, _ = provider.compute_batch_weights({"index": torch.tensor([11, 10])})
            torch.testing.assert_close(weights, torch.tensor([1.5, 0]))
            with self.assertRaises(KeyError):
                provider.compute_batch_weights({"index": torch.tensor([9])})
            with self.assertRaises(KeyError):
                provider.compute_batch_weights({})

    def test_duplicate_or_nan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.parquet"
            for ids, values in (([0, 0], [1.0, 1.0]), ([0, 1], [1.0, np.nan]), ([0, 1], [1.0, -1.0])):
                pd.DataFrame({"index": ids, "graph_transition_weight": values}).to_parquet(path)
                with self.assertRaises(ValueError):
                    GRBCWeights(path)

    def test_zero_batch_no_optimizer_or_scheduler_update(self):
        from lerobot.scripts.lerobot_train import update_policy

        policy = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        provider = SimpleNamespace(compute_batch_weights=lambda batch: (torch.zeros(2), {}))
        accelerator = SimpleNamespace(reduce=lambda x, reduction: x)
        tracker = SimpleNamespace()
        _, logs = update_policy(
            tracker,
            policy,
            {},
            optimizer,
            1.0,
            accelerator,
            lr_scheduler=scheduler,
            grbc_weights_provider=provider,
        )
        self.assertEqual(logs["grbc_skipped_all_zero"], 1)
        self.assertEqual(scheduler.last_epoch, 0)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_real_trainer_weighted_backward(self):
        from lerobot.scripts.lerobot_train import update_policy

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, batch, reduction="mean"):
                return self.parameter * torch.tensor([1.0, 3.0]), {}

        policy = Policy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        provider = SimpleNamespace(
            reduction="mean", compute_batch_weights=lambda batch: (torch.tensor([0.0, 1.0]), {})
        )
        accelerator = SimpleNamespace(
            reduce=lambda x, reduction: x,
            autocast=nullcontext,
            num_processes=1,
            backward=lambda loss: loss.backward(),
            unwrap_model=lambda model, **kwargs: model,
        )
        tracker = SimpleNamespace()
        update_policy(tracker, policy, {}, optimizer, 0.0, accelerator, grbc_weights_provider=provider)
        self.assertAlmostEqual(policy.parameter.item(), 0.85, places=6)


class ModelContractTest(unittest.TestCase):
    def test_smolvla_state_metadata_matches_target_dataset(self):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.factory import make_policy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        class Policy(torch.nn.Module):
            def __init__(self, config, **kwargs):
                super().__init__()
                self.config = config

        def build(state_dim):
            cfg = SmolVLAConfig(
                device="cpu",
                input_features={
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                },
            )
            meta = SimpleNamespace(
                features={
                    "observation.state": {"dtype": "float32", "shape": [state_dim]},
                    "action": {"dtype": "float32", "shape": [8]},
                },
                stats={},
            )
            with patch("lerobot.policies.factory.get_policy_class", return_value=Policy):
                return make_policy(cfg, ds_meta=meta)

        self.assertEqual(tuple(build(8).config.input_features["observation.state"].shape), (8,))
        with self.assertRaisesRegex(ValueError, "max_state_dim"):
            build(33)

    def test_three_backbones_expose_per_sample_loss(self):
        # Exercise the bundled forward bodies with stubbed model internals,
        # without downloading large VLA checkpoints or importing incompatible Transformers.
        root = Path(__file__).resolve().parents[2] / "src/lerobot/policies"
        batch = {
            "observation.language.tokens": torch.ones(2, 3, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 3, dtype=torch.bool),
        }
        for name, cls in (("smolvla", "SmolVLAPolicy"), ("xvla", "XVLAPolicy"), ("pi05", "PI05Policy")):
            code = (root / name / f"modeling_{name}.py").read_text()
            tree = ast.parse(code)
            klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
            method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
            module = ast.Module(body=[method], type_ignores=[])
            ns = {
                "Tensor": torch.Tensor,
                "torch": torch,
                "OBS_STATE": "observation.state",
                "ACTION": "action",
                "OBS_LANGUAGE_TOKENS": "observation.language.tokens",
                "OBS_LANGUAGE_ATTENTION_MASK": "observation.language.attention_mask",
            }
            exec(compile(module, f"{name}_forward", "exec"), ns)
            fake = SimpleNamespace(
                config=SimpleNamespace(
                    adapt_to_pi_aloha=False,
                    max_action_dim=3,
                    output_features={"action": SimpleNamespace(shape=(3,))},
                ),
                prepare_images=lambda batch: (None, None),
                prepare_state=lambda batch: None,
                prepare_action=lambda batch: None,
                _preprocess_images=lambda batch: (None, None),
                _build_model_inputs=lambda batch: {},
                _prepare_action_targets=lambda batch: None,
            )
            if name == "xvla":
                fake.model = lambda **kwargs: {"action_loss": torch.tensor([1.0, 3.0])}
            else:
                fake.model = SimpleNamespace(forward=lambda *args: torch.ones(2, 4, 3))
            loss, _ = ns["forward"](fake, batch, reduction="none")
            self.assertEqual(loss.shape, (2,), name)


class ConfigTest(unittest.TestCase):
    def test_presets_match_published_json(self):
        root = Path(__file__).resolve().parents[2]
        for name, preset in PRESETS.items():
            config = GraphConfig(**{**asdict(GraphConfig()), **preset})
            config.validate()
            given = json.loads((root / "configs/grbc" / f"{name}.json").read_text())
            self.assertEqual({key: asdict(config)[key] for key in given}, given)

    def test_invalid_configs(self):
        for override in (
            {"stride": 0},
            {"history": -1},
            {"n_action_step": 31},
            {"adaptive_base_scale": 0},
            {"weight_min": 2},
            {"weight_gamma": float("nan")},
            {"interpolation": "foo"},
        ):
            with self.assertRaises(ValueError):
                GraphConfig(**override).validate()


if __name__ == "__main__":
    unittest.main()
