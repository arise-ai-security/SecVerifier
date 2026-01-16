#!/usr/bin/env python3
"""Build missing SEC-bench Docker images concurrently.

This script builds Docker images for SEC-bench vulnerability instances that are
not available on Docker Hub. It follows the official SEC-bench image building
process from https://github.com/SEC-bench/SEC-bench.

The generated images are compatible with the SEC-bench evaluation framework
and include the `secb` helper script for build/repro/patch commands.
"""

import os
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# Images to build
ALL_IDS = [
    "exiv2.cve-2017-17669",
    "gpac.cve-2023-0770",
    "imagemagick.cve-2017-12641",
    "mruby.cve-2022-0240",
    "njs.cve-2022-28049",
    "njs.cve-2022-32414",
    "cjson.cve-2016-10749",
    "exiv2.cve-2017-11339",
    "faad2.cve-2021-32273",
    "faad2.cve-2021-32276",
    "flac.cve-2020-22219",
    "gpac.cve-2023-2838",
    "imagemagick.cve-2017-11754",
    "libarchive.cve-2016-10209",
    "libarchive.cve-2017-14501",
    "libiec61850.cve-2018-19122",
    "libjpeg-turbo.cve-2020-17541",
    "libsass.cve-2018-20822",
    "libtorrent.cve-2016-7164",
    "mruby.cve-2022-1071",
    "openjpeg.cve-2017-14164",
]

# Set to specific IDs to retry, or None/[] to build all
RETRY_IDS = []

# Use RETRY_IDS if set, otherwise ALL_IDS
MISSING_IDS = RETRY_IDS if RETRY_IDS else ALL_IDS

# Number of concurrent builds (2 recommended for macOS to avoid memory pressure)
MAX_WORKERS = 2


def get_project_name_for_oss_fuzz(repo: str) -> str:
    """Get the project name for OSS-Fuzz compatibility.

    This matches the official SEC-bench implementation.
    """
    # Split by '/' and take the last part
    name = repo.split("/")[-1]
    # Replace underscores with hyphens
    name = name.replace("_", "-")
    # Special case conversions
    if name == "php-src":
        name = "php"
    return name


def get_secb_script(instance_id: str, work_dir: str) -> str:
    """Generate the secb helper script for an instance.

    This matches the official SEC-bench template from:
    https://github.com/SEC-bench/SEC-bench/blob/main/secb/preprocessor/templates/secb_helper.sh.j2

    Args:
        instance_id: The vulnerability instance ID
        work_dir: The working directory in the container
    """
    return f'''#!/bin/bash
set -euo pipefail

build() {{
    echo "BUILDING THE PROJECT..."

    # Handle git sub-modules
    if [[ -f .gitmodules || -f .gitmodule ]]; then
        echo "Detected git sub-modules - initialising/updating..."
        git submodule update --init --recursive
    else
        echo "No git sub-modules found - skipping update."
    fi

    # Check for repo_changes.diff and apply if it exists and hasn't been applied yet
    if [[ -f /testcase/repo_changes.diff ]]; then
        # Check if the patch has already been applied to avoid re-applying
        if ! git apply --check /testcase/repo_changes.diff &>/dev/null; then
            echo "Repository changes already applied or cannot be applied cleanly. Proceeding with build."
        else
            echo "Applying repository changes from repo_changes.diff..."
            git apply /testcase/repo_changes.diff || echo "Warning: Could not apply repo_changes.diff cleanly. Proceeding anyway."
        fi
    fi

    # Use /usr/local/bin/compile which applies sanitizer flags and runs $SRC/build.sh
    # stdout: /dev/null
    # stderr: grep filters out "warning:" and lets everything else through
    if /usr/local/bin/compile \\
         1>/dev/null \\
         2> >(grep -Fv --line-buffered -e "warning:" -e "SyntaxWarning:" -e "WARNING:" >&2); then
        echo "BUILD COMPLETED SUCCESSFULLY!"
    else
        echo "BUILD FAILED!"
        exit 1
    fi
}}

repro() {{
    echo "REPRODUCING THE ISSUE FOR {instance_id}..."
    # TODO: Add commands to trigger the specific vulnerability
    # For now, it's a placeholder.
    echo "PLACEHOLDER: TRIGGER VULNERABILITY HERE."
    # NOTE: YOU SHOULD NOT RETURN/EXIT 0 IN THIS FUNCTION.
    # Example: /out/fuzzer /testcase/poc_file
}}

patch() {{
    echo "PATCHING THE PROJECT..."
    cd {work_dir}

    # Check for repo_changes.diff and apply if it exists and hasn't been applied yet
    if [[ -f /testcase/repo_changes.diff ]]; then
        # Check if the patch has already been applied to avoid re-applying
        if ! git apply --check /testcase/repo_changes.diff &>/dev/null; then
            echo "Repository changes already applied or cannot be applied cleanly. Proceeding with patch."
        else
            echo "Applying repository changes from repo_changes.diff..."
            git apply /testcase/repo_changes.diff || echo "Warning: Could not apply repo_changes.diff cleanly. Proceeding anyway."
        fi
    fi

    if git apply /testcase/model_patch.diff; then
        echo "PATCH APPLIED SUCCESSFULLY!"
    else
        echo "PATCH APPLICATION FAILED!"
        exit 1
    fi
}}

if [ "$#" -ge 1 ]; then
    command="$1"

    case "$command" in
        build)
            build "$@"
            ;;
        repro)
            repro "$@"
            ;;
        patch)
            patch "$@"
            ;;
        *)
            echo "Unknown command: $command"
            echo "Usage: secb [build|repro|patch]"
            exit 1
            ;;
    esac
else
    echo "Usage: secb [build|repro|patch]"
    exit 1
fi
'''


def get_dockerfile_additions(
    script_name: str,
    sanitizer: str,
    lang: str,
    project_name: str,
    work_dir: str,
) -> str:
    """Generate Dockerfile additions following official SEC-bench template.

    This matches the official SEC-bench template from:
    https://github.com/SEC-bench/SEC-bench/blob/main/secb/preprocessor/templates/Dockerfile.instance.j2
    https://github.com/SEC-bench/SEC-bench/blob/main/secb/evaluator/templates/Dockerfile.eval.instance.j2

    Args:
        script_name: Name of the helper script (secb)
        sanitizer: Sanitizer type (address, memory, undefined, etc.)
        lang: Programming language (c, c++, etc.)
        project_name: Project name for OSS-Fuzz compatibility
        work_dir: Working directory in the container
    """
    return f'''
# Copy helper script
COPY {script_name} /usr/local/bin/
RUN chmod +x /usr/local/bin/{script_name}

# Create testcase directory (standard practice)
RUN mkdir -p /testcase

# Environment variables (matches official SEC-bench)
ENV SANITIZER="{sanitizer}"
ENV FUZZING_ENGINE=libfuzzer
ENV FUZZING_LANGUAGE="{lang or 'c++'}"
ENV PROJECT_NAME="{project_name}"
ENV ARCHITECTURE=x86_64
ENV LANG="C.UTF-8"
ENV CFLAGS="-w -Wno-yacc -Wno-incompatible-pointer-types"
ENV CXXFLAGS="-w -Wno-yacc -Wno-incompatible-pointer-types"

# Workspace directory
WORKDIR {work_dir}
'''


def build_image(instance: dict) -> tuple[str, bool, str]:
    """Build a single Docker image from instance data.

    Follows the official SEC-bench image building process.

    Returns: (instance_id, success, message)
    """
    instance_id = instance['instance_id']
    repo = instance.get('repo', '')
    work_dir = instance['work_dir']
    sanitizer = instance.get('sanitizer', 'address')
    lang = instance.get('lang', 'c++')
    tag = f"hwiwonlee/secb.eval.x86_64.{instance_id}:latest"

    # Get project name for OSS-Fuzz compatibility
    project_name = get_project_name_for_oss_fuzz(repo)

    # Create temp directory for build context
    build_dir = tempfile.mkdtemp(prefix=f"secb_build_{instance_id}_")

    try:
        # Get Dockerfile content from dataset
        dockerfile_content = instance['dockerfile']

        # Patch: ensure apt-get update runs before apt-get install
        # This fixes stale package list issues
        dockerfile_content = dockerfile_content.replace(
            'RUN apt-get install',
            'RUN apt-get update && apt-get install'
        )

        # Add SEC-bench specific additions (secb script, env vars, etc.)
        dockerfile_content += get_dockerfile_additions(
            script_name='secb',
            sanitizer=sanitizer,
            lang=lang,
            project_name=project_name,
            work_dir=work_dir,
        )

        dockerfile_path = os.path.join(build_dir, "Dockerfile")
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        # Write build.sh (required by Dockerfile and /usr/local/bin/compile)
        if instance.get('build_sh'):
            build_sh_path = os.path.join(build_dir, "build.sh")
            with open(build_sh_path, 'w') as f:
                f.write(instance['build_sh'])

        # Write secb helper script (matches official SEC-bench template)
        secb_path = os.path.join(build_dir, "secb")
        with open(secb_path, 'w') as f:
            f.write(get_secb_script(instance_id, work_dir))

        # Write additional files if present (list of {filename, content} dicts)
        if instance.get('additional_files'):
            for file_info in instance['additional_files']:
                filename = file_info.get('filename')
                content = file_info.get('content')
                if filename and content is not None:
                    filepath = os.path.join(build_dir, filename)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    with open(filepath, 'w') as f:
                        f.write(content)

        print(f"[STARTED] Building {instance_id} (sanitizer: {sanitizer}, lang: {lang})...")

        # Build image (--no-cache to avoid stale layer issues)
        result = subprocess.run(
            [
                "docker", "build",
                "--no-cache",
                "--platform=linux/amd64",
                "-t", tag,
                build_dir
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            print(f"[SUCCESS] {instance_id}")
            return (instance_id, True, "Build successful")
        else:
            print(f"[FAILED] {instance_id}")
            # Save error log
            error_log = f"build_error_{instance_id}.log"
            with open(error_log, 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            return (instance_id, False, f"Build failed. See {error_log}")

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {instance_id}")
        return (instance_id, False, "Build timed out after 1 hour")
    except Exception as e:
        print(f"[ERROR] {instance_id}: {e}")
        return (instance_id, False, str(e))
    finally:
        # Cleanup temp directory
        shutil.rmtree(build_dir, ignore_errors=True)


def main():
    print("Loading SEC-bench/Seed dataset...")
    ds = load_dataset("SEC-bench/Seed", split="cve")

    # Filter to only missing instances
    instances_to_build = []
    for instance in ds:
        if instance['instance_id'] in MISSING_IDS:
            instances_to_build.append(instance)

    print(f"Found {len(instances_to_build)} instances to build")
    print(f"Building with {MAX_WORKERS} concurrent workers\n")

    results = {"success": [], "failed": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(build_image, inst): inst['instance_id']
            for inst in instances_to_build
        }

        for future in as_completed(futures):
            instance_id, success, message = future.result()
            if success:
                results["success"].append(instance_id)
            else:
                results["failed"].append((instance_id, message))

    # Print summary
    print("\n" + "=" * 50)
    print("BUILD SUMMARY")
    print("=" * 50)
    print(f"Successful: {len(results['success'])}")
    for iid in results['success']:
        print(f"  + {iid}")

    print(f"\nFailed: {len(results['failed'])}")
    for iid, msg in results['failed']:
        print(f"  - {iid}: {msg}")


if __name__ == "__main__":
    main()
