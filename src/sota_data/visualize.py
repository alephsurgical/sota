"""Visualize SOTA datasets using Rerun."""

import argparse
import time
from pathlib import Path

import numpy as np
import rerun as rr

from sota_data.load import (
    get_episode_frames,
    get_frame_subtask,
    get_subtask_annotations,
    list_datasets,
    load_dataset,
    load_subtask_labels,
)


def visualize_episode(
    dataset_id: str,
    episode_index: int = 0,
    root: str = "data",
    fps: float = 50.0,
    cameras: list[str] | None = None,
    show_state: bool = True,
    show_action: bool = True,
):
    """Visualize an episode with Rerun.

    Args:
        dataset_id: Dataset identifier (e.g., "twoarm-anastomosis-real-dagger-200/20260311_170113")
        episode_index: Episode to visualize
        root: Data root directory
        fps: Playback framerate
        cameras: Camera views to display (default: all)
        show_state: Show robot state bar chart
        show_action: Show action bar chart

    Controls:
        - Use Rerun viewer controls for playback
        - Press Ctrl+C in terminal to stop
    """
    # Initialize Rerun
    rr.init("SOTA Episode Viewer")
    rr.spawn(memory_limit="10%")

    # Load dataset
    print(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, root)
    start_idx, end_idx = get_episode_frames(dataset, episode_index)
    num_frames = end_idx - start_idx

    # Get camera keys
    camera_keys = dataset.meta.camera_keys
    if cameras:
        camera_keys = [f"observation.images.{c}" for c in cameras]
    print(f"Cameras: {[k.split('.')[-1] for k in camera_keys]}")

    # Load annotations
    annotations = get_subtask_annotations(dataset_id, root)
    try:
        subtask_labels = load_subtask_labels(dataset_id.split("/")[0], root)
    except FileNotFoundError:
        subtask_labels = []

    print(f"Playing episode {episode_index}: {num_frames} frames at {fps} fps")
    print("Press Ctrl+C to stop")

    frame_duration = 1.0 / fps

    try:
        for frame_offset in range(num_frames):
            frame_idx = start_idx + frame_offset
            row = dataset[frame_idx]

            # Set timeline
            rr.set_time_sequence("frame", frame_offset)
            rr.set_time_seconds("time", frame_offset / fps)

            # Log camera frames
            for cam_key in camera_keys:
                if cam_key in row:
                    frame = row[cam_key]
                    if hasattr(frame, "numpy"):
                        frame = frame.numpy()
                    # CHW -> HWC if needed
                    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                        frame = frame.transpose(1, 2, 0)

                    cam_name = cam_key.split(".")[-1]
                    rr.log(f"camera/{cam_name}", rr.Image(frame))

            # Get subtask label
            subtask = get_frame_subtask(annotations, episode_index, frame_offset)
            if not subtask:
                subtask = "unlabeled"

            # Log info panel
            progress = f"{frame_offset + 1}/{num_frames}"
            info_text = f"## Frame: {progress}\n\n## Subtask: {subtask}"
            rr.log("info", rr.TextDocument(info_text, media_type=rr.MediaType.MARKDOWN))

            # Log state
            if show_state and "observation.state" in row:
                state = row["observation.state"]
                if hasattr(state, "numpy"):
                    state = state.numpy()
                rr.log("state", rr.BarChart(state.astype(np.float32)))

            # Log action
            if show_action and "action" in row:
                action = row["action"]
                if hasattr(action, "numpy"):
                    action = action.numpy()
                rr.log("action", rr.BarChart(action.astype(np.float32)))

            time.sleep(frame_duration)

    except KeyboardInterrupt:
        print("\nStopped.")

    print("Playback complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SOTA episodes with Rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available datasets
    sota-visualize --list

    # Visualize first episode with all cameras
    sota-visualize twoarm-anastomosis-real-dagger-200/20260311_170113

    # Visualize episode 5 with only TOP_VIEW camera
    sota-visualize DATASET_ID --episode 5 --cameras TOP_VIEW

    # Slow playback at 10 fps
    sota-visualize DATASET_ID --fps 10
        """
    )
    parser.add_argument("dataset_id", nargs="?", help="Dataset ID to visualize")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (default: 0)")
    parser.add_argument("--cameras", nargs="+",
                       choices=["TOP_VIEW", "BACK_VIEW", "WRIST_LEFT", "WRIST_RIGHT"],
                       help="Camera views to display (default: all)")
    parser.add_argument("--fps", type=float, default=50.0, help="Playback FPS (default: 50)")
    parser.add_argument("--root", default="data", help="Data root directory")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--no-state", action="store_true", help="Hide state bar chart")
    parser.add_argument("--no-action", action="store_true", help="Hide action bar chart")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:\n")
        datasets = list_datasets(args.root)
        for ds in datasets:
            print(f"  {ds['id']}")
            print(f"    Episodes: {ds['num_episodes']}, Frames: {ds['total_frames']}")
        return

    if not args.dataset_id:
        parser.print_help()
        print("\nError: dataset_id is required (or use --list)")
        return

    visualize_episode(
        args.dataset_id,
        args.episode,
        args.root,
        args.fps,
        args.cameras,
        show_state=not args.no_state,
        show_action=not args.no_action,
    )


if __name__ == "__main__":
    main()
