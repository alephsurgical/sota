# Data Format

SOTA datasets use the [LeRobot](https://github.com/huggingface/lerobot) v3.0 format, a standard for robotics datasets supporting multimodal observations and efficient storage.

## Directory Structure

```
data/
└── twoarm-anastomosis-real-dagger-200/     # Dataset root
    ├── subtask_labels.json                  # Shared label vocabulary
    └── 20260311_170113/                     # Session (timestamp-based ID)
        ├── meta/
        │   ├── info.json                    # Dataset metadata
        │   ├── stats.json                   # Feature statistics (min/max/mean/std)
        │   ├── episodes/
        │   │   └── chunk-000/
        │   │       └── file-000.parquet     # Episode metadata
        │   ├── tasks.parquet                # Task index mapping
        │   ├── subtasks.parquet             # Subtask index mapping
        │   └── subtask_annotations.json     # Frame-level annotations
        ├── data/
        │   └── chunk-000/
        │       └── file-000.parquet         # State/action data
        └── videos/
            ├── observation.images.WRIST_LEFT/
            ├── observation.images.WRIST_RIGHT/
            ├── observation.images.TOP_VIEW/
            └── observation.images.BACK_VIEW/
                └── chunk-000/
                    └── file-000.mp4         # Video files
```

## Robot Configuration

**Robot Type:** Dual xArm6 arms (`xarm-xarm`)

### State Vector (38 dimensions)

The `observation.state` contains the full robot state:

| Indices | Description | Dim |
|---------|-------------|-----|
| 0-2 | Left arm position (x, y, z) | 3 |
| 3-5 | Left arm orientation (roll, pitch, yaw) | 3 |
| 6 | Left gripper position | 1 |
| 7-12 | Left arm joint positions | 6 |
| 13-18 | Left arm joint torques | 6 |
| 19-21 | Right arm position (x, y, z) | 3 |
| 22-24 | Right arm orientation (roll, pitch, yaw) | 3 |
| 25 | Right gripper position | 1 |
| 26-31 | Right arm joint positions | 6 |
| 32-37 | Right arm joint torques | 6 |

### Action Vector (26 dimensions)

The `action` contains the commanded motion:

| Indices | Description | Dim |
|---------|-------------|-----|
| 0-5 | Left arm delta (dx, dy, dz, droll, dpitch, dyaw) | 6 |
| 6 | Left gripper command | 1 |
| 7-12 | Left arm joint targets | 6 |
| 13-18 | Right arm delta (dx, dy, dz, droll, dpitch, dyaw) | 6 |
| 19 | Right gripper command | 1 |
| 20-25 | Right arm joint targets | 6 |

## Camera Configuration

| Camera | Resolution | Description |
|--------|------------|-------------|
| `WRIST_LEFT` | 1920x1080 | Left arm wrist camera |
| `WRIST_RIGHT` | 1920x1080 | Right arm wrist camera |
| `TOP_VIEW` | 1280x720 | Overhead camera |
| `BACK_VIEW` | 1280x720 | Back-angle camera |

All cameras record at **50 fps**.

## Data Files

### meta/info.json

Dataset configuration and schema:

```json
{
  "codebase_version": "v3.0",
  "robot_type": "xarm-xarm",
  "total_episodes": 52,
  "total_frames": 10559,
  "fps": 50,
  "chunks_size": 1000,
  "features": {
    "observation.state": {"dtype": "float32", "shape": [38]},
    "action": {"dtype": "float32", "shape": [26]},
    "observation.images.WRIST_LEFT": {"video": true, "shape": [3, 1080, 1920]},
    ...
  }
}
```

### data/chunk-XXX/file-XXX.parquet

Frame data with columns:

| Column | Type | Description |
|--------|------|-------------|
| `observation.state` | float32[38] | Robot state vector |
| `action` | float32[26] | Action vector |
| `timestamp` | float32 | Frame timestamp (seconds) |
| `frame_index` | int64 | Frame index within episode |
| `episode_index` | int64 | Episode ID |
| `index` | int64 | Global frame index |
| `task_index` | int64 | Task ID (usually 0) |
| `subtask_index` | int64 | Subtask ID (maps to subtask_labels.json) |
| `success` | bool | Episode success flag |

### meta/episodes/chunk-XXX/file-XXX.parquet

Episode metadata with columns:

| Column | Description |
|--------|-------------|
| `episode_index` | Episode ID |
| `length` | Number of frames |
| `dataset_from_index` | Start frame in global index |
| `dataset_to_index` | End frame in global index |
| `videos/*/chunk_index` | Video chunk for each camera |
| `videos/*/file_index` | Video file for each camera |
| `videos/*/from_timestamp` | Video start time |
| `videos/*/to_timestamp` | Video end time |

### meta/stats.json

Feature statistics for normalization:

```json
{
  "observation.state": {
    "min": [...],
    "max": [...],
    "mean": [...],
    "std": [...]
  },
  "action": {...}
}
```

## Loading Data

```python
from sota_data import load_dataset, iterate_episode_frames

# Load a session
dataset = load_dataset("twoarm-anastomosis-real-dagger-200/20260311_170113")

# Access by global index
frame = dataset[0]
state = frame["observation.state"]  # torch.Tensor (38,)
action = frame["action"]            # torch.Tensor (26,)
wrist_left = frame["observation.images.WRIST_LEFT"]  # torch.Tensor (3, 1080, 1920)

# Iterate by episode
for frame in iterate_episode_frames(dataset, episode_index=0):
    # Process frame...
    pass

# Normalization
import json
with open("data/twoarm-anastomosis-real-dagger-200/20260311_170113/meta/stats.json") as f:
    stats = json.load(f)

state_mean = torch.tensor(stats["observation.state"]["mean"])
state_std = torch.tensor(stats["observation.state"]["std"])
normalized_state = (state - state_mean) / state_std
```

## Video Decoding

Videos are stored as H.264 MP4 files. The LeRobot library handles decoding transparently:

```python
dataset = load_dataset(dataset_id, video_backend="pyav")  # or "torchvision"

# Videos are decoded on-demand when accessed
frame = dataset[0]
wrist_left = frame["observation.images.WRIST_LEFT"]
```

For faster iteration without video decoding, download metadata only:

```bash
python scripts/download_data.py --metadata-only
```

Then load without image keys:

```python
# Access only state/action (no video decoding)
state = dataset[0]["observation.state"]
action = dataset[0]["action"]
```
