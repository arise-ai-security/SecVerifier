"""Manages Docker image building for SEC-Bench instances."""

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class InstanceDockerManager:
    """Builds and manages instance-specific Docker images with auto-build support."""

    # Custom base image (replaces hwiwonlee/secb.base:latest)
    CUSTOM_BASE_IMAGE = "arise-lab-security/secb.base:test"
    PREBUILT_IMAGE_PREFIX = "hwiwonlee/secb.x86_64."

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        platform: str = "linux/amd64",
        auto_build: bool = True,
    ):
        """Initialize the Docker manager with automatic base image building.

        Args:
            cache_dir: Directory to cache built images (optional)
            platform: Docker platform (default: linux/amd64)
            auto_build: If True, automatically build base image if missing (default: True)
        """
        self.cache_dir = cache_dir
        self.platform = platform
        self.auto_build = auto_build

        # Auto-build base image if enabled and missing
        if auto_build:
            self._auto_build_base_image()

    def _auto_build_base_image(self):
        """Automatically build the custom base image if it doesn't exist.

        This method is called during initialization to ensure the base image
        is always available. On first run, it builds the image from docker/Dockerfile.base.
        Subsequent runs skip this step as the image is cached.
        """
        # Check if base image already exists
        if self._image_exists_locally(self.CUSTOM_BASE_IMAGE):
            logger.info(f"✅ Base image found: {self.CUSTOM_BASE_IMAGE}")
            return

        # Image doesn't exist - build it automatically
        print(f"\n🔍 Checking base image: {self.CUSTOM_BASE_IMAGE}")
        print(f"⚠️  Base image not found. Building automatically...")
        print(f"🏗️  Building {self.CUSTOM_BASE_IMAGE}")
        print(f"⏱️  This takes ~30-60 minutes on first run (one-time only)")
        print(f"☕ You can get coffee while it builds...\n")

        # Locate Dockerfile.base
        # Try to find it relative to the current file location
        current_file = Path(__file__).resolve()
        repo_root = current_file.parent.parent.parent.parent  # Navigate up to repo root
        dockerfile_path = repo_root / "docker" / "Dockerfile.base"

        if not dockerfile_path.exists():
            # Fallback: try current working directory
            dockerfile_path = Path.cwd() / "docker" / "Dockerfile.base"

        if not dockerfile_path.exists():
            error_msg = (
                f"Cannot find Dockerfile.base at {dockerfile_path}\n"
                f"Expected location: <repo>/docker/Dockerfile.base\n"
                f"Please ensure the Dockerfile.base file exists."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        build_context = dockerfile_path.parent
        logger.info(f"Found Dockerfile at: {dockerfile_path}")
        logger.info(f"Build context: {build_context}")

        try:
            # Build the base image
            cmd = [
                "docker", "build",
                "--platform", self.platform,
                "-t", self.CUSTOM_BASE_IMAGE,
                "-f", str(dockerfile_path),
                str(build_context)
            ]

            logger.info(f"Running: {' '.join(cmd)}")

            # Run build with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in process.stdout:
                # Print build progress to console
                sys.stdout.write(line)
                sys.stdout.flush()

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Docker build failed with exit code {process.returncode}")

            print(f"\n✅ Base image built successfully: {self.CUSTOM_BASE_IMAGE}")
            print(f"📦 Image is cached and will be reused in future runs\n")

        except subprocess.SubprocessError as e:
            error_msg = f"Failed to build base image: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during base image build: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def get_instance_image(self, instance_data: dict) -> str:
        """Get or build the Docker image for an instance.

        This method tries three approaches in order:
        1. Check for pre-built image on Docker Hub
        2. Check local cache
        3. Build from dataset Dockerfile

        Args:
            instance_data: Dataset instance containing dockerfile, build_sh, etc.

        Returns:
            Image name/tag to use with DockerWorkspace
        """
        instance_id = instance_data['instance_id']

        # Option 1: Try pre-built image
        prebuilt_image = f"{self.PREBUILT_IMAGE_PREFIX}{instance_id}:latest"
        if self._try_pull_prebuilt(prebuilt_image):
            return prebuilt_image

        # Option 2: Check local cache
        local_image = f"secb-instance:{instance_id}"
        if self._image_exists_locally(local_image):
            logger.info(f"Using cached image: {local_image}")
            return local_image

        # Option 3: Build from dataset Dockerfile
        logger.info(f"Building instance image from dataset: {instance_id}")
        return self._build_instance_image(instance_data)

    def ensure_agent_server_image(self, base_image: str, instance_id: str) -> str:
        """Build a minimal OpenHands agent-server image on top of the base image.

        This avoids OpenHands' monorepo build by installing the server via pip.

        Args:
            base_image: The base instance image (e.g., `secb-instance:<id>`)
            instance_id: Dataset instance id

        Returns:
            The built agent-server image tag
        """
        image_tag = f"secb-agent-server:{instance_id}"

        # Reuse if present
        if self._image_exists_locally(image_tag):
            logger.info(f"Using cached agent-server image: {image_tag}")
            return image_tag

        logger.info(f"Building agent-server image from base: {base_image}")

        dockerfile = f"""
ARG BASE_IMAGE
FROM ${{BASE_IMAGE}}

# Ensure Python and pip are available
RUN set -eux; \
    if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl wget git; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install OpenHands agent server (pulls required dependencies)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir openhands-agent-server

EXPOSE 8000
ENTRYPOINT ["python3", "-m", "openhands.agent_server"]
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            ctx = Path(temp_dir)
            (ctx / "Dockerfile").write_text(dockerfile)
            try:
                subprocess.run(
                    [
                        "docker", "build",
                        "--platform", self.platform,
                        "--build-arg", f"BASE_IMAGE={base_image}",
                        "-t", image_tag,
                        str(ctx),
                    ],
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Successfully built agent-server image: {image_tag}")
                return image_tag
            except subprocess.CalledProcessError as e:
                err = e.stderr.decode() if e.stderr else str(e)
                logger.error(f"Failed to build agent-server image: {err}")
                raise RuntimeError(
                    f"Failed to build agent-server image for {instance_id}: {err}"
                )

    def _try_pull_prebuilt(self, image_name: str) -> bool:
        """Try to pull a pre-built image from Docker Hub.

        Args:
            image_name: Full image name with tag

        Returns:
            True if successfully pulled, False otherwise
        """
        try:
            logger.info(f"Checking for pre-built image: {image_name}")
            result = subprocess.run(
                ["docker", "pull", image_name],
                capture_output=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode == 0:
                logger.info(f"Successfully pulled pre-built image: {image_name}")
                return True
            return False
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.debug(f"Pre-built image not available: {e}")
            return False

    def _image_exists_locally(self, image_name: str) -> bool:
        """Check if an image exists in local Docker cache.

        Args:
            image_name: Image name to check

        Returns:
            True if image exists locally
        """
        result = subprocess.run(
            ["docker", "images", "-q", image_name],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())

    def _build_instance_image(self, instance_data: dict) -> str:
        """Build an instance-specific Docker image from dataset Dockerfile.

        Args:
            instance_data: Dataset instance data

        Returns:
            Built image name/tag
        """
        instance_id = instance_data['instance_id']
        dockerfile_content = instance_data['dockerfile']
        build_sh_content = instance_data['build_sh']

        # Fix the base image tag in Dockerfile
        dockerfile_content = self._fix_base_image_tag(dockerfile_content)

        # Create temporary build context
        with tempfile.TemporaryDirectory() as temp_dir:
            build_context = Path(temp_dir)

            # Write Dockerfile
            (build_context / 'Dockerfile').write_text(dockerfile_content)

            # Write build.sh
            (build_context / 'build.sh').write_text(build_sh_content)

            # Build the image
            image_tag = f"secb-instance:{instance_id}"
            logger.info(f"Building Docker image: {image_tag}")

            try:
                subprocess.run(
                    [
                        "docker", "build",
                        "--platform", self.platform,
                        "-t", image_tag,
                        str(build_context)
                    ],
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Successfully built image: {image_tag}")
                return image_tag

            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                logger.error(f"Failed to build image: {error_msg}")
                raise RuntimeError(
                    f"Failed to build Docker image for {instance_id}: {error_msg}"
                )

    def _fix_base_image_tag(self, dockerfile_content: str) -> str:
        """Replace hwiwonlee/secb.base:latest with arise-lab-security/secb.base:test.

        Transforms:
            FROM hwiwonlee/secb.base:latest
        Into:
            FROM arise-lab-security/secb.base:test

        Args:
            dockerfile_content: Original Dockerfile content from dataset

        Returns:
            Modified Dockerfile content with custom base image
        """
        pattern = r'FROM\s+hwiwonlee/secb\.base:latest'
        replacement = f'FROM {self.CUSTOM_BASE_IMAGE}'

        fixed_content = re.sub(pattern, replacement, dockerfile_content)

        if fixed_content != dockerfile_content:
            logger.info(f"Replaced base image: hwiwonlee/secb.base:latest → {self.CUSTOM_BASE_IMAGE}")

        return fixed_content

    def cleanup_instance_image(self, instance_id: str) -> None:
        """Remove an instance image from local Docker cache.

        Args:
            instance_id: Instance ID
        """
        image_tag = f"secb-instance:{instance_id}"
        try:
            subprocess.run(
                ["docker", "rmi", image_tag],
                capture_output=True,
                check=True,
            )
            logger.info(f"Removed image: {image_tag}")
        except subprocess.CalledProcessError:
            logger.debug(f"Image not found or already removed: {image_tag}")
