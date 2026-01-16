#!/usr/bin/env python3
"""Build missing SEC-bench INSTANCE Docker images using SEC-bench preprocessor.

This script is a wrapper around SEC-bench's build_instance_images.py that:
1. Loads instance data from the SEC-bench/Seed HuggingFace dataset
2. Filters to specified instance IDs
3. Builds INSTANCE images (hwiwonlee/secb.x86_64.{instance_id}:latest)

These INSTANCE images use the original OSS-Fuzz Dockerfiles with proper
sanitizer support via /usr/local/bin/compile.

Usage:
    python build_instance_images.py --ids njs.cve-2022-32414 gpac.cve-2023-0770
    python build_instance_images.py --all
    python build_instance_images.py --missing  # Only build images not on Docker Hub
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from datasets import load_dataset


# SEC-bench repository path (adjust if needed)
SEC_BENCH_PATH = Path.home() / "data" / "github" / "SEC-bench"


def docker_hub_image_exists(namespace: str, repo: str) -> bool:
    """Check if a repository exists on Docker Hub."""
    url = f"https://hub.docker.com/v2/repositories/{namespace}/{repo}/tags/latest"
    try:
        response = httpx.get(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def get_missing_instance_ids(instance_ids: list[str]) -> list[str]:
    """Filter to only instance IDs that don't have images on Docker Hub."""
    missing = []
    for instance_id in instance_ids:
        repo_name = f"secb.x86_64.{instance_id}"
        if not docker_hub_image_exists("hwiwonlee", repo_name):
            print(f"  [MISSING] {instance_id}")
            missing.append(instance_id)
        else:
            print(f"  [EXISTS]  {instance_id}")
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Build INSTANCE Docker images for SEC-bench using the preprocessor."
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific instance IDs to build",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all instances from the dataset",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Only build instances that are missing from Docker Hub",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--dataset",
        default="SEC-bench/Seed",
        help="HuggingFace dataset name (default: SEC-bench/Seed)",
    )
    parser.add_argument(
        "--split",
        default="cve",
        help="Dataset split to use (default: cve)",
    )
    parser.add_argument(
        "--sec-bench-path",
        type=Path,
        default=SEC_BENCH_PATH,
        help=f"Path to SEC-bench repository (default: {SEC_BENCH_PATH})",
    )

    args = parser.parse_args()

    if not args.ids and not args.all and not args.missing:
        parser.error("Please specify --ids, --all, or --missing")

    # Verify SEC-bench path
    build_script = args.sec_bench_path / "secb" / "preprocessor" / "build_instance_images.py"
    if not build_script.exists():
        print(f"ERROR: SEC-bench preprocessor not found at {build_script}")
        print(f"Please check --sec-bench-path or set SEC_BENCH_PATH correctly")
        sys.exit(1)

    # Load dataset
    print(f"Loading dataset: {args.dataset} (split: {args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} instances")

    # Get all instance IDs from dataset
    all_instance_ids = [inst["instance_id"] for inst in ds]

    # Determine which instances to build
    if args.ids:
        target_ids = args.ids
        # Validate that all specified IDs exist in dataset
        missing_from_dataset = set(target_ids) - set(all_instance_ids)
        if missing_from_dataset:
            print(f"WARNING: These instance IDs not found in dataset: {missing_from_dataset}")
            target_ids = [i for i in target_ids if i in all_instance_ids]
    elif args.all:
        target_ids = all_instance_ids
    else:
        target_ids = all_instance_ids  # Will be filtered below

    # Filter to missing images if requested
    if args.missing:
        print("\nChecking Docker Hub for existing images...")
        target_ids = get_missing_instance_ids(target_ids)

    if not target_ids:
        print("\nNo instances to build!")
        sys.exit(0)

    print(f"\nWill build {len(target_ids)} INSTANCE images:")
    for iid in target_ids[:10]:
        print(f"  - hwiwonlee/secb.x86_64.{iid}:latest")
    if len(target_ids) > 10:
        print(f"  ... and {len(target_ids) - 10} more")

    # Filter dataset to target instances
    instances_to_build = [inst for inst in ds if inst["instance_id"] in target_ids]

    # Create temporary JSONL file for SEC-bench preprocessor
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="secb_instances_"
    ) as f:
        for instance in instances_to_build:
            # Convert to dict and handle any non-serializable types
            inst_dict = dict(instance)
            json.dump(inst_dict, f)
            f.write("\n")
        jsonl_path = f.name

    print(f"\nCreated temporary JSONL file: {jsonl_path}")

    # Run SEC-bench build_instance_images.py
    cmd = [
        sys.executable,
        "-m",
        "secb.preprocessor.build_instance_images",
        "--input-file",
        jsonl_path,
        "--workers",
        str(args.workers),
    ]

    print(f"\nRunning SEC-bench preprocessor...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(args.sec_bench_path),
            check=False,
        )
        if result.returncode != 0:
            print(f"\nBuild completed with return code: {result.returncode}")
        else:
            print(f"\nBuild completed successfully!")
    finally:
        # Clean up temp file
        Path(jsonl_path).unlink(missing_ok=True)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
