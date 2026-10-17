"""Frozen MCR ResNet50 feature extraction with validated caching."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import dataset_fingerprint, decode_requested, video_requests


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_features(
    root, nodes, episodes, info, cameras, checkpoint, cache, batch_size=48, device="cuda", reuse=False
):
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet50

    checkpoint, cache = Path(checkpoint), Path(cache)
    if not checkpoint.is_file():
        raise FileNotFoundError("Provide the official MCR ResNet50 checkpoint with --mcr-ckpt")
    identity = {
        "dataset": dataset_fingerprint(root),
        "checkpoint_sha256": sha256_file(checkpoint),
        "cameras": cameras,
        "preprocess": "resize256-centercrop224-totensor-no-imagenet-norm-v1",
        "nodes": nodes[["index", "episode_index", "frame_index"]].values.tolist(),
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"mcr_{signature}.npy"
    if reuse and path.exists():
        features = np.load(path, allow_pickle=False)
        if features.shape != (len(nodes), 6144) or not np.isfinite(features).all():
            raise ValueError("Incomplete or invalid feature cache")
        return features, {
            "cache_hit": True,
            "fingerprint": signature,
            "checkpoint_sha256": identity["checkpoint_sha256"],
        }
    if batch_size < 1:
        raise ValueError("batch-size-mcr must be positive")
    model = resnet50(weights=None)
    model.fc = torch.nn.Identity()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False).to(device)
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    camera_features = []

    def flush(pending, positions, result):
        if not pending:
            return
        with torch.inference_mode():
            encoded = model(torch.stack(pending).to(device)).float().cpu().numpy()
        for vector, ids in zip(encoded, positions, strict=True):
            result[ids] = vector
        pending.clear()
        positions.clear()

    for camera in cameras:
        result = np.empty((len(nodes), 2048), dtype=np.float32)
        pending, positions = [], []

        for video, requests in video_requests(root, nodes, episodes, info, camera).items():
            print(f"Encoding {camera}: {video.name}, {len(requests)} frames", flush=True)
            for ids, rgb in decode_requested(video, requests):
                pending.append(transform(Image.fromarray(rgb)))
                positions.append(ids)
                if len(pending) >= batch_size:
                    flush(pending, positions, result)
        flush(pending, positions, result)
        camera_features.append(result)
    features = np.concatenate(camera_features, axis=1)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite encoded features")
    temp = path.with_suffix(".tmp.npy")
    np.save(temp, features)
    temp.replace(path)
    return features, {
        "cache_hit": False,
        "fingerprint": signature,
        "checkpoint_sha256": identity["checkpoint_sha256"],
    }
