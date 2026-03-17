# SOTA - Surgical Robotics Training Data

Training data for learning surgical manipulation policies on dual-arm robotic systems. This dataset contains demonstrations of the **anastomosis** task (suturing blood vessels together) with frame-level subtask annotations.

## Quick Start

### 1. Install

```bash
git clone https://github.com/alephsurgical/sota.git
cd sota
pip install -e .
```

### 2. Configure R2 Access

Install rclone and add the SOTA remote:

```bash
# macOS
brew install rclone

# Linux
curl https://rclone.org/install.sh | sudo bash
```

Add to `~/.config/rclone/rclone.conf`:

```ini
[sota]
type = s3
provider = Cloudflare
access_key_id = <ACCESS_KEY>
secret_access_key = <SECRET_KEY>
endpoint = https://b16f4a3487a8546e6a300d1a7494c334.r2.cloudflarestorage.com
```

### 3. Download Data

```bash
# List available datasets
python scripts/download_data.py --list

# Download all data (~100GB with videos)
python scripts/download_data.py

# Download without videos (~10GB, faster)
python scripts/download_data.py --metadata-only

# Download specific dataset
python scripts/download_data.py --dataset twoarm-anastomosis-real-optim-00
```

### 4. Load and Explore

```python
from sota_data import load_dataset, load_subtask_labels

# Load a dataset session
dataset = load_dataset("twoarm-anastomosis-real-optim-00/20260218_134852")
print(f"Episodes: {dataset.num_episodes}, Frames: {len(dataset)}")

# Access a frame
frame = dataset[0]
state = frame["observation.state"]   # Robot state (38,)
action = frame["action"]             # Action (26,)
wrist_left = frame["observation.images.WRIST_LEFT"]  # Camera (3, 1080, 1920)

# Load subtask labels
labels = load_subtask_labels("twoarm-anastomosis-real-optim-00")
print(labels)
# ['grasp needle body with driver', 'align needle to vessel axis with forceps', ...]
```

### 5. Visualize

```bash
# Visualize with Rerun
python examples/visualize_episode.py

# Or directly:
python -m sota_data.visualize twoarm-anastomosis-real-optim-00/20260218_134852
```

## Data Overview

| Feature | Description |
|---------|-------------|
| **Robot** | Dual xArm6 arms |
| **Task** | Anastomosis (vessel suturing) |
| **State dim** | 38 (positions, orientations, joints, torques) |
| **Action dim** | 26 (deltas + joint targets) |
| **Cameras** | 4 views (2x wrist 1080p, 1x overhead 720p, 1x back 720p) |
| **FPS** | 50 Hz |
| **Annotations** | 17 surgical subtasks per frame |

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for detailed schema.

See [docs/SUBTASK_LABELS.md](docs/SUBTASK_LABELS.md) for subtask definitions.

## File Structure

```
data/
└── twoarm-anastomosis-*/
    ├── subtask_labels.json         # Subtask vocabulary (19 labels)
    └── {session_id}/
        ├── meta/
        │   ├── info.json           # Dataset metadata
        │   ├── stats.json          # Normalization stats
        │   ├── episodes/           # Episode boundaries
        │   └── subtask_annotations.json
        ├── data/                   # Parquet files (state, action)
        └── videos/                 # MP4 files per camera
```

## API Reference

```python
from sota_data import (
    load_dataset,           # Load a dataset session
    load_all_datasets,      # Load all available sessions
    load_subtask_labels,    # Load subtask vocabulary
    get_subtask_annotations,# Load frame-level annotations
    iterate_episode_frames, # Iterate frames in an episode
)
```

## License

[License details]
