import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.policies.grbc.config import GraphConfig
from lerobot.policies.grbc.dataset import (
    CAMERAS,
    decode_requested,
    load_dataset,
    sample_nodes,
    table_identity,
    video_requests,
)


def fixture(root):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "total_frames": 6,
        "total_episodes": 2,
        "fps": 10,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {camera: {"dtype": "video", "info": {"video.fps": 10}} for camera in CAMERAS},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for ep in range(2):
        data = pd.DataFrame(
            {
                "index": np.arange(3 * ep, 3 * ep + 3),
                "episode_index": ep,
                "frame_index": np.arange(3),
                "timestamp": np.arange(3) / 10,
                "observation.state": [np.array([0.0, 1.0])] * 3,
                "action": [np.array([0.0, 0.0])] * 3,
                "next.reward": [0.0, 1.0, 0.0] if ep == 0 else [0.0, 0.0, 0.0],
            }
        )
        data.to_parquet(root / f"data/chunk-000/file-{ep:03d}.parquet")
        episode = {"episode_index": ep}
        for camera in CAMERAS:
            episode.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": ep * 0.3,
                }
            )
        pd.DataFrame([episode]).to_parquet(root / f"meta/episodes/chunk-000/file-{ep:03d}.parquet")
    return info


class DatasetTest(unittest.TestCase):
    def test_all_shards_and_forced_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            data, episodes, _ = load_dataset(root)
            self.assertEqual(len(data), 6)
            self.assertEqual(len(episodes), 2)
            nodes = sample_nodes(data, GraphConfig(stride=15))
            self.assertEqual(nodes[nodes.episode_index == 0].frame_index.tolist(), [0, 1, 2])
            self.assertEqual(nodes[nodes.episode_index == 1].frame_index.tolist(), [0, 2])

    def test_missing_rewards_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "data/chunk-000/file-001.parquet"
            data = pd.read_parquet(path).drop(columns="next.reward")
            data.to_parquet(path)
            with self.assertRaises(ValueError):
                load_dataset(root)

    def test_duplicate_index_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "data/chunk-000/file-001.parquet"
            data = pd.read_parquet(path)
            data["index"] = 0
            data.to_parquet(path)
            with self.assertRaises(ValueError):
                load_dataset(root)

    def test_identity_changes_with_actions_not_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            data, _, _ = load_dataset(root)
            identity = table_identity(data)
            self.assertEqual(identity, table_identity(data.iloc[::-1]))
            data.at[0, "action"] = np.array([1.0, 0.0])
            self.assertNotEqual(identity, table_identity(data))

    def test_packed_video_offset_and_actual_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = fixture(root)
            data, episodes, _ = load_dataset(root)
            camera = CAMERAS[0]
            path = root / info["video_path"].format(video_key=camera, chunk_index=0, file_index=0)
            path.parent.mkdir(parents=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width = 32
                stream.height = 32
                stream.pix_fmt = "yuv420p"
                for i in range(6):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((32, 32, 3), i * 30, dtype=np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            rows = data[data.episode_index == 1].iloc[:1]
            requests = video_requests(root, rows, episodes, info, camera)
            self.assertEqual(list(requests[path]), [3])
            decoded = list(decode_requested(path, requests[path]))
            self.assertEqual(len(decoded), 1)
            self.assertAlmostEqual(decoded[0][1].mean(), 90, delta=6)
            with self.assertRaises(ValueError):
                list(decode_requested(path, {100: [0]}))


if __name__ == "__main__":
    unittest.main()
