# Dataset Contract

Prepare data as **LeRobot v3.0** (`meta/info.json: codebase_version = "v3.0"`). Dataset conversion, real-robot collection, and deployment are intentionally outside this release.

```text
task_v3/
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  data/chunk-000/file-000.parquet
  videos/observation.images.cam_0/chunk-000/file-000.mp4
  videos/observation.images.cam_1/chunk-000/file-000.mp4
  videos/observation.images.cam_2/chunk-000/file-000.mp4
```

All parquet shards under `data/` and `meta/episodes/` are read, not just file 000. Standard LeRobot metadata and statistics are needed for VLA training even though the graph needs fewer fields.

## Required Frame Fields

| Field | Requirement |
|---|---|
| `index` | Unique original global integer sample ID |
| `episode_index` | Episode ID matching episode metadata |
| `frame_index` | Contiguous zero-based local frame ID |
| `timestamp` | Finite episode-local seconds, increasing with frames |
| `observation.state` | Robot state vector including a gripper measurement |
| `action` | Policy training action vector in the dataset's original representation |
| `next.reward` | Genuine sparse outcome annotation; positive at actual success frames |
| Three RGB camera features | Video features named `observation.images.cam_0`, `_1`, `_2` by default |

`next.done` and `complementary_info.is_intervention` are optional for graph generation. Intervention flags default to false; no human label is fabricated. Missing reward labels are an error. The exporter never infers success merely from trajectory termination and never marks all episodes successful. A dataset must contain at least one labeled success.

Customize camera names with `--cameras VIEW_A VIEW_B VIEW_C`. Episode metadata must contain `videos/<camera>/chunk_index`, `file_index`, and `from_timestamp` for each view. Video addressing adds the episode's packed-video `from_timestamp` to the frame's local timestamp. This matters when multiple episodes share one video file.

## Gripper Convention

The default reads `observation.state[-1] > 0.5` as a binary gripper category. This is a dataset convention, not a universal physical-open/closed definition. Change `--gripper-state-index` and `--gripper-threshold` for your data. Verify the units and separation of open/closed readings before relying on the hard edge filter. A continuous aperture in millimeters must not silently use a threshold intended for normalized [0, 1] values.

## Export / Training Alignment

Train on exactly the same frame indexing used to generate `transition_weights.parquet`. Each sampled training item is an action chunk beginning at global `index`; its single graph weight multiplies the policy's full per-sample loss. This is not an independent weight for each predicted action coordinate.

The trainer rejects missing or duplicate weight IDs and nonfinite/negative values. When exporter metadata is present, it also checks a portable table-content identity. Copying data to another machine is supported; filtering, reindexing, action/state changes, or reward edits require regenerating weights. Feature cache validation is stricter and includes file signatures; moving datasets can invalidate a cache without invalidating the exported labels.

Construct a separate graph per task. Mixing unrelated task goals into one graph can create semantically invalid shortcuts. Avoid including validation episodes in a training graph when reporting held-out progress accuracy. Use the separate isolated-query evaluator described in EVALUATION.md rather than treating full-dataset graph progress as held-out predictions.
