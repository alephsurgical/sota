#!/usr/bin/env python3
"""Download SOTA datasets from Cloudflare R2.

Usage:
    python scripts/download_data.py                      # Download all datasets
    python scripts/download_data.py --dataset NAME       # Download specific dataset
    python scripts/download_data.py --list               # List available datasets
    python scripts/download_data.py --metadata-only      # Download only metadata (no videos)
"""

import argparse
import subprocess
import sys
from pathlib import Path

R2_REMOTE = "sota"
R2_BUCKET = "sota"
LOCAL_ROOT = Path("data")


def run_rclone(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run an rclone command."""
    cmd = ["rclone"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def check_rclone_config() -> bool:
    """Check if rclone is configured with the sota remote."""
    result = run_rclone(["listremotes"], check=False)
    if result.returncode != 0:
        print("Error: rclone not found. Please install it first.")
        print("  macOS: brew install rclone")
        print("  Linux: curl https://rclone.org/install.sh | sudo bash")
        return False

    if f"{R2_REMOTE}:" not in result.stdout:
        print(f"Error: '{R2_REMOTE}' remote not configured in rclone.")
        print("Run: bash scripts/setup_rclone.sh")
        return False

    return True


def list_datasets() -> list[str]:
    """List all available datasets in the bucket."""
    result = run_rclone(["lsd", f"{R2_REMOTE}:{R2_BUCKET}"], check=False)
    if result.returncode != 0:
        print(f"Error listing datasets: {result.stderr}")
        return []

    datasets = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            parts = line.split()
            if parts:
                datasets.append(parts[-1])
    return datasets


def download_dataset(
    name: str,
    force: bool = False,
    metadata_only: bool = False,
) -> bool:
    """Download a dataset."""
    remote_path = f"{R2_REMOTE}:{R2_BUCKET}/{name}"
    local_path = LOCAL_ROOT / name

    if local_path.exists() and not force:
        print(f"Skipping {name} (already exists). Use --force to re-download.")
        return True

    local_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name}...")

    if metadata_only:
        # Download only metadata files (no videos)
        cmd = [
            "copy", remote_path, str(local_path),
            "--progress",
            "--transfers", "8",
            "--include", "subtask_labels.json",
            "--include", "**/meta/**",
            "--include", "**/data/**",
        ]
    else:
        # Download everything
        cmd = [
            "copy", remote_path, str(local_path),
            "--progress",
            "--transfers", "8",
        ]

    result = run_rclone(cmd, check=False)

    if result.returncode != 0:
        print(f"Error downloading {name}: {result.stderr}")
        return False

    print(f"Downloaded to: {local_path.absolute()}")
    return True


def get_dataset_size(name: str) -> str:
    """Get the size of a dataset."""
    result = run_rclone(
        ["size", f"{R2_REMOTE}:{R2_BUCKET}/{name}", "--json"],
        check=False
    )
    if result.returncode == 0:
        import json
        try:
            data = json.loads(result.stdout)
            bytes_val = data.get("bytes", 0)
            if bytes_val > 1e9:
                return f"{bytes_val / 1e9:.1f} GB"
            elif bytes_val > 1e6:
                return f"{bytes_val / 1e6:.1f} MB"
            else:
                return f"{bytes_val / 1e3:.1f} KB"
        except json.JSONDecodeError:
            pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Download SOTA datasets from R2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_data.py --list
    python scripts/download_data.py --dataset twoarm-anastomosis-real-dagger-200
    python scripts/download_data.py --metadata-only  # Skip videos (~80% smaller)
    python scripts/download_data.py  # Download everything
        """
    )
    parser.add_argument("--dataset", type=str, help="Specific dataset to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--metadata-only", action="store_true",
                       help="Download only metadata and state/action data (no videos)")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    global LOCAL_ROOT
    LOCAL_ROOT = Path(args.output)

    if not check_rclone_config():
        sys.exit(1)

    if args.list:
        print("Available datasets:\n")
        datasets = list_datasets()
        for ds in datasets:
            size = get_dataset_size(ds)
            print(f"  {ds}  ({size})")
        print(f"\nTotal: {len(datasets)} datasets")
        return

    if args.dataset:
        success = download_dataset(args.dataset, args.force, args.metadata_only)
        sys.exit(0 if success else 1)
    else:
        print("Downloading all datasets...")
        datasets = list_datasets()

        if not datasets:
            print("No datasets found.")
            sys.exit(1)

        failed = []
        for ds in datasets:
            if not download_dataset(ds, args.force, args.metadata_only):
                failed.append(ds)

        if failed:
            print(f"\nFailed to download: {', '.join(failed)}")
            sys.exit(1)

        print(f"\nDone! Downloaded {len(datasets)} datasets to: {LOCAL_ROOT.absolute()}")


if __name__ == "__main__":
    main()
