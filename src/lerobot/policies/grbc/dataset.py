"""Validated, all-shard LeRobot v3.0 tables and packed-video addressing."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAMERAS = [f"observation.images.cam_{i}" for i in range(3)]


def scalar(value):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar column, got shape {array.shape}")
    return array.reshape(-1)[0].item()


def load_dataset(root, cameras=CAMERAS):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Prepare a LeRobot v3.0 dataset before graph construction")

    def read_shards(directory):
        paths = sorted((root / directory).rglob("*.parquet"))
        if not paths:
            raise ValueError(f"No parquet shards in {directory}")
        return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    data, episodes = read_shards("data"), read_shards("meta/episodes")
    required = {
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
        "next.reward",
    }
    if required - set(data):
        raise ValueError(f"Missing columns: {sorted(required - set(data))}. Supply genuine success rewards.")
    for key in ("index", "episode_index", "frame_index", "timestamp", "next.reward"):
        data[key] = data[key].map(scalar)
        if not np.isfinite(data[key].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {key}")
    for key in ("index", "episode_index", "frame_index"):
        if not (data[key] == data[key].astype(np.int64)).all():
            raise ValueError(f"Nonintegral {key}")
        data[key] = data[key].astype(np.int64)
    for key in ("next.done", "complementary_info.is_intervention"):
        data[key] = data[key].map(scalar).astype(bool) if key in data else False
    if data["index"].duplicated().any() or data.duplicated(["episode_index", "frame_index"]).any():
        raise ValueError("Duplicate global index or episode/frame pair")
    if episodes["episode_index"].duplicated().any():
        raise ValueError("Duplicate episode metadata")
    if len(data) != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Dataset metadata counts do not match all parquet shards")
    if set(data.episode_index) != set(episodes.episode_index):
        raise ValueError("Frame and episode metadata IDs disagree")
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    for episode, rows in data.groupby("episode_index"):
        if not np.array_equal(rows.frame_index.to_numpy(), np.arange(len(rows))):
            raise ValueError(f"Episode {episode} must have contiguous zero-based frame_index")
        if np.any(np.diff(rows.timestamp.to_numpy()) < 0):
            raise ValueError(f"Episode {episode} has nonmonotonic timestamps")
    for camera in cameras:
        if info["features"].get(camera, {}).get("dtype") != "video":
            raise ValueError(f"Expected video feature {camera}")
        for field in ("chunk_index", "file_index", "from_timestamp"):
            if f"videos/{camera}/{field}" not in episodes:
                raise ValueError(f"Missing packed-video metadata for {camera}: {field}")
    return data, episodes, info


def sample_nodes(data, config):
    parts = []
    for _, rows in data.groupby("episode_index", sort=True):
        rows = rows.sort_values("frame_index").copy()
        selected = set(range(0, len(rows), config.stride)) | {len(rows) - 1}
        selected.update(np.flatnonzero(rows["next.reward"].to_numpy() > 0))
        if config.include_intervention_boundaries:
            flags = rows["complementary_info.is_intervention"].to_numpy(dtype=int)
            selected.update(np.flatnonzero(np.diff(flags, prepend=flags[0])))
        part = rows.iloc[sorted(selected)].copy()
        part["episode_success"] = bool((rows["next.reward"] > 0).any())
        part["episode_length"] = len(rows)
        parts.append(part)
    nodes = pd.concat(parts, ignore_index=True)
    nodes["node_id"] = np.arange(len(nodes))
    nodes["global_index"] = nodes["index"]
    nodes["next_reward"] = nodes["next.reward"]
    values = nodes["observation.state"].map(lambda state: np.asarray(state)[config.gripper_state_index])
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite gripper state")
    nodes["gripper_closed"] = values > config.gripper_threshold
    return nodes


def history_rows(nodes, history):
    mapping = np.empty((len(nodes), history + 1), dtype=np.int64)
    for _, rows in nodes.groupby("episode_index", sort=True):
        ids = rows.sort_values("frame_index").node_id.to_numpy(dtype=int)
        for position, node_id in enumerate(ids):
            mapping[node_id] = ids[np.maximum(0, position - np.arange(history + 1))]
    return mapping


def video_requests(root, rows, episodes, info, camera):
    metadata = episodes.set_index("episode_index")
    requests = {}
    for position, row in enumerate(rows.itertuples(index=False)):
        meta = metadata.loc[row.episode_index]
        path = Path(root) / info["video_path"].format(
            video_key=camera,
            chunk_index=int(meta[f"videos/{camera}/chunk_index"]),
            file_index=int(meta[f"videos/{camera}/file_index"]),
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        fps = float(info["features"][camera].get("info", {}).get("video.fps", info["fps"]))
        timestamp = float(meta[f"videos/{camera}/from_timestamp"]) + float(row.timestamp)
        target = int(round(timestamp * fps))
        requests.setdefault(path, {}).setdefault(target, []).append(position)
    return requests


def decode_requested(path, requests):
    """Yield requested absolute video frames using PTS, not episode-local offsets."""
    import av

    remaining = set(requests)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate)
        start = stream.start_time or 0
        first = min(remaining)
        container.seek(
            start + int(first / fps / float(stream.time_base)), stream=stream, backward=True, any_frame=False
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            index = int(round((frame.pts - start) * float(stream.time_base) * fps))
            if index in remaining:
                remaining.remove(index)
                yield requests[index], frame.to_rgb().to_ndarray()
                if not remaining:
                    break
    if remaining:
        raise ValueError(f"Missing frames in {path.name}: {sorted(remaining)[:10]}")


def dataset_fingerprint(root):
    root = Path(root)
    records = []
    for folder, suffix in (("data", "*.parquet"), ("meta", "*.parquet"), ("videos", "*.mp4")):
        for path in sorted((root / folder).rglob(suffix)):
            stat = path.stat()
            records.append((str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns))
    payload = {"info": json.loads((root / "meta/info.json").read_text()), "files": records}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def table_identity(data):
    """Portable frame identity; unlike the cache key this survives copying machines."""
    digest = hashlib.sha256()
    ordered = data.sort_values("index")
    for key in (
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "next.reward",
        "observation.state",
        "action",
    ):
        if key in ordered:
            values = np.asarray(ordered[key].tolist(), dtype="<f8")
            digest.update(key.encode())
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
    return digest.hexdigest()
