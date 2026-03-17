"""Load SOTA datasets using the LeRobot library."""

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Default data directory
DEFAULT_ROOT = Path("data")

# Index keys
DATASET_FROM_INDEX = "dataset_from_index"
DATASET_TO_INDEX = "dataset_to_index"


def load_dataset(
    dataset_id: str,
    root: Path | str = DEFAULT_ROOT,
    video_backend: str = "pyav",
) -> LeRobotDataset:
    """Load a SOTA dataset.

    Args:
        dataset_id: Dataset identifier (e.g., "twoarm-anastomosis-real-dagger-200/20260311_170113")
        root: Root data directory (default: "data")
        video_backend: Video decoding backend ("pyav" or "torchvision")

    Returns:
        LeRobotDataset instance

    Example:
        >>> dataset = load_dataset("twoarm-anastomosis-real-dagger-200/20260311_170113")
        >>> print(f"Episodes: {dataset.num_episodes}, Frames: {len(dataset)}")
        >>> frame = dataset[0]
        >>> state = frame["observation.state"]  # Shape: (38,)
        >>> action = frame["action"]  # Shape: (26,)
    """
    root = Path(root)
    dataset_path = root / dataset_id

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Run: python scripts/download_data.py --dataset {dataset_id.split('/')[0]}"
        )

    return LeRobotDataset(
        repo_id=dataset_id,
        root=dataset_path,
        video_backend=video_backend,
        revision="main",  # Use 'main' to avoid HuggingFace version checks for local data
    )


def load_all_datasets(
    root: Path | str = DEFAULT_ROOT,
    video_backend: str = "pyav",
) -> list[LeRobotDataset]:
    """Load all available SOTA datasets.

    Args:
        root: Root data directory
        video_backend: Video decoding backend

    Returns:
        List of LeRobotDataset instances
    """
    root = Path(root)
    datasets = []

    for dataset_root in sorted(root.iterdir()):
        if not dataset_root.is_dir():
            continue
        # Each dataset_root (e.g., twoarm-anastomosis-real-dagger-200) contains sessions
        for session_dir in sorted(dataset_root.iterdir()):
            info_path = session_dir / "meta" / "info.json"
            if info_path.exists():
                dataset_id = f"{dataset_root.name}/{session_dir.name}"
                try:
                    ds = load_dataset(dataset_id, root, video_backend)
                    datasets.append(ds)
                except Exception as e:
                    print(f"Warning: Failed to load {dataset_id}: {e}")

    return datasets


def list_datasets(root: Path | str = DEFAULT_ROOT) -> list[dict]:
    """List all available datasets with metadata.

    Args:
        root: Root data directory

    Returns:
        List of dicts with dataset info (id, num_episodes, fps, etc.)
    """
    root = Path(root)
    datasets = []

    for dataset_root in sorted(root.iterdir()):
        if not dataset_root.is_dir():
            continue

        for session_dir in sorted(dataset_root.iterdir()):
            info_path = session_dir / "meta" / "info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                datasets.append({
                    "id": f"{dataset_root.name}/{session_dir.name}",
                    "num_episodes": info.get("total_episodes", 0),
                    "total_frames": info.get("total_frames", 0),
                    "fps": info.get("fps", 50),
                })

    return datasets


def load_subtask_labels(
    dataset_name: str = "twoarm-anastomosis-real-dagger-200",
    root: Path | str = DEFAULT_ROOT,
) -> list[str]:
    """Load the shared subtask label vocabulary.

    Args:
        dataset_name: Dataset root name containing subtask_labels.json
        root: Root data directory

    Returns:
        List of subtask label strings
    """
    root = Path(root)
    labels_path = root / dataset_name / "subtask_labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            "Make sure to download a dataset first."
        )

    with open(labels_path) as f:
        data = json.load(f)

    return data.get("labels", [])


def get_subtask_annotations(
    dataset_id: str,
    root: Path | str = DEFAULT_ROOT,
) -> dict | None:
    """Load subtask annotations for a dataset session.

    Args:
        dataset_id: Dataset identifier (e.g., "twoarm-anastomosis-real-dagger-200/20260311_170113")
        root: Root data directory

    Returns:
        Annotation dict with structure:
        {
            "version": "1.0",
            "episodes": {
                "0": {
                    "segments": [
                        {"start_frame": 0, "end_frame": 15, "label": "unused", ...},
                        ...
                    ]
                },
                ...
            }
        }
    """
    root = Path(root)
    ann_path = root / dataset_id / "meta" / "subtask_annotations.json"

    if not ann_path.exists():
        return None

    with open(ann_path) as f:
        return json.load(f)


def get_episode_frames(
    dataset: LeRobotDataset,
    episode_index: int,
) -> tuple[int, int]:
    """Get the frame range for an episode.

    Args:
        dataset: LeRobotDataset instance
        episode_index: Episode index (0-based)

    Returns:
        Tuple of (start_frame_idx, end_frame_idx)
    """
    ep = dataset.meta.episodes[episode_index]
    return ep[DATASET_FROM_INDEX], ep[DATASET_TO_INDEX]


def iterate_episode_frames(
    dataset: LeRobotDataset,
    episode_index: int,
) -> Iterator[dict]:
    """Iterate over all frames in an episode.

    Args:
        dataset: LeRobotDataset instance
        episode_index: Episode index

    Yields:
        Frame dictionaries containing:
        - observation.state: Robot state (38,)
        - action: Action vector (26,)
        - observation.images.*: Camera images
        - episode_index, frame_index, timestamp, etc.

    Example:
        >>> for frame in iterate_episode_frames(dataset, episode_index=0):
        ...     state = frame["observation.state"]
        ...     action = frame["action"]
        ...     # Process frame...
    """
    start_idx, end_idx = get_episode_frames(dataset, episode_index)
    for frame_idx in range(start_idx, end_idx):
        yield dataset[frame_idx]


def get_frame_subtask(
    annotations: dict,
    episode_index: int,
    frame_offset: int,
) -> str | None:
    """Get the subtask label for a specific frame.

    Args:
        annotations: Annotation dict from get_subtask_annotations()
        episode_index: Episode index
        frame_offset: Frame offset within the episode (0-based)

    Returns:
        Subtask label string, or None if not annotated
    """
    if not annotations:
        return None

    episode_key = str(episode_index)
    if episode_key not in annotations.get("episodes", {}):
        return None

    segments = annotations["episodes"][episode_key].get("segments", [])
    for seg in segments:
        if seg["start_frame"] <= frame_offset < seg["end_frame"]:
            return seg["label"]

    return None
