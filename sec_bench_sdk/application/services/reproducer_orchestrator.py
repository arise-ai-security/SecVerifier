"""Main orchestrator service that coordinates the multi-phase workflow."""

from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from sec_bench_sdk.application.dto.phase_data import (
    PhaseInput,
    PhaseOutput,
    ReproducerInput,
    ReproducerOutput,
)
from sec_bench_sdk.application.services.phase_executor import PhaseExecutor
from sec_bench_sdk.domain.value_objects import AgentType, DELEGATION_SEQUENCE, PhaseConfig
from sec_bench_sdk.infrastructure.sdk.docker_manager import InstanceDockerManager
from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMFactory, LLMConfig
from sec_bench_sdk.infrastructure.sdk.conversation_runner import detect_platform

logger = logging.getLogger(__name__)


class ReproducerOrchestrator:
    """Orchestrates the 3-phase vulnerability reproduction workflow.

    This replaces the Reproducer agent logic from the original implementation.
    It coordinates:
    1. BuilderAgent - validates and fixes build scripts
    2. ExploiterAgent - creates proof-of-concept exploits
    3. FixerAgent - creates patches to fix vulnerabilities

    Each phase runs sequentially, with retry logic if needed.
    """

    def __init__(
        self,
        phase_executor: PhaseExecutor,
        llm_config: LLMConfig,
        max_retries: int = 1,
        host_port: int = 8010,
        extra_ports: bool = False,
    ):
        """Initialize the orchestrator.

        Args:
            phase_executor: Service for executing individual phases
            llm_config: LLM configuration
            max_retries: Maximum retries per phase
            host_port: Port for the Docker workspace
            extra_ports: Enable VS Code and VNC access
        """
        self.phase_executor = phase_executor
        self.llm_config = llm_config
        self.max_retries = max_retries
        self.phase_configs = PhaseConfig.get_default_configs()
        self.host_port = host_port
        self.extra_ports = extra_ports

        # Initialize Docker manager
        self.docker_manager = InstanceDockerManager(
            platform=detect_platform(),
            auto_build=True,
        )

    async def execute(self, input_data: ReproducerInput) -> ReproducerOutput:
        """Execute the complete 3-phase workflow.

        CRITICAL: Uses a SINGLE DockerWorkspace for all phases to maintain state.
        This ensures that files created by BuilderAgent are available to ExploiterAgent,
        and modifications by ExploiterAgent are visible to FixerAgent.

        Args:
            input_data: Input data for reproduction

        Returns:
            ReproducerOutput with results from all phases
        """
        # Import DockerWorkspace here to avoid module-level import issues
        from openhands.workspace import DockerWorkspace

        output = ReproducerOutput(instance_id=input_data.instance_id)

        try:
            # Get instance-specific Docker image (builds if necessary)
            logger.info(f"[{input_data.instance_id}] Getting Docker image...")
            instance_image = self.docker_manager.get_instance_image(input_data.metadata)
            logger.info(f"[{input_data.instance_id}] Using image: {instance_image}")

            # Create ONE workspace for all phases
            # This is CRITICAL - state must persist across phases
            logger.info(f"[{input_data.instance_id}] Creating Docker workspace...")
            with DockerWorkspace(
                base_image=instance_image,
                host_port=self.host_port,
                extra_ports=self.extra_ports,
                platform=detect_platform(),
            ) as workspace:
                logger.info(f"[{input_data.instance_id}] Workspace created successfully")

                # Initialize workspace (clone repo, setup directories)
                logger.info(f"[{input_data.instance_id}] Initializing workspace...")
                self._setup_workspace(workspace, input_data)
                logger.info(f"[{input_data.instance_id}] Workspace initialized")

                # Phase 1: Builder
                logger.info(f"[{input_data.instance_id}] Starting Builder phase...")
                builder_output = await self._execute_phase_with_retry(
                    AgentType.BUILDER,
                    input_data,
                    workspace,
                    previous_output=None,
                )
                output.builder_output = builder_output

                if not builder_output.success:
                    output.success = False
                    output.error_message = f"Builder phase failed: {builder_output.error_message}"
                    return output

                logger.info(f"[{input_data.instance_id}] Builder phase completed successfully")

                # Phase 2: Exploiter
                logger.info(f"[{input_data.instance_id}] Starting Exploiter phase...")
                exploiter_output = await self._execute_phase_with_retry(
                    AgentType.EXPLOITER,
                    input_data,
                    workspace,
                    previous_output=builder_output,
                )
                output.exploiter_output = exploiter_output

                if not exploiter_output.success:
                    output.success = False
                    output.error_message = f"Exploiter phase failed: {exploiter_output.error_message}"
                    return output

                logger.info(f"[{input_data.instance_id}] Exploiter phase completed successfully")

                # Phase 3: Fixer
                logger.info(f"[{input_data.instance_id}] Starting Fixer phase...")
                fixer_output = await self._execute_phase_with_retry(
                    AgentType.FIXER,
                    input_data,
                    workspace,
                    previous_output=exploiter_output,
                )
                output.fixer_output = fixer_output

                if not fixer_output.success:
                    output.success = False
                    output.error_message = f"Fixer phase failed: {fixer_output.error_message}"
                    return output

                logger.info(f"[{input_data.instance_id}] Fixer phase completed successfully")

                # All phases succeeded
                output.success = True
                logger.info(f"[{input_data.instance_id}] All phases completed successfully!")
                return output

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"[{input_data.instance_id}] Error during execution: {error_details}")
            output.success = False
            output.error_message = f"Orchestrator error: {str(e)}\n{error_details}"
            return output

    def _setup_workspace(self, workspace: 'DockerWorkspace', input_data: ReproducerInput):
        """Initialize the workspace before agent execution.

        This sets up the environment that all agents will work in:
        1. Clones the repository at the base commit
        2. Creates necessary directories (/testcase, etc.)
        3. Writes initial files (build.sh, secb script, etc.)

        Args:
            workspace: The Docker workspace to initialize
            input_data: Input data with repo URL, commit, etc.
        """
        try:
            logger.info("Setting up workspace directories...")

            # Create /testcase directory for exploits and artifacts
            workspace.execute_command("mkdir -p /testcase && chmod 777 /testcase")

            # Clone repository to /src/repo
            repo_url = input_data.repository_url
            base_commit = input_data.base_commit

            logger.info(f"Cloning repository: {repo_url}")
            clone_result = workspace.execute_command(
                f"git clone --depth 1 {repo_url} /src/repo || git clone {repo_url} /src/repo"
            )

            if clone_result.returncode != 0:
                logger.warning(f"Clone failed with shallow history, trying full clone...")
                workspace.execute_command(f"git clone {repo_url} /src/repo")

            # Checkout base commit
            logger.info(f"Checking out base commit: {base_commit}")
            workspace.execute_command(f"cd /src/repo && git checkout {base_commit}")

            # Write initial build.sh if provided in metadata
            if 'build_sh' in input_data.metadata:
                logger.info("Writing initial build script...")
                build_sh_content = input_data.metadata['build_sh']
                # Write to a temporary file and then move it
                workspace.execute_command(
                    f"cat > /src/build.sh << 'EOFBUILDSH'\n{build_sh_content}\nEOFBUILDSH"
                )
                workspace.execute_command("chmod +x /src/build.sh")

            # Write secb utility script (used by agents)
            self._write_secb_script(workspace, input_data)

            logger.info("Workspace setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup workspace: {e}")
            raise

    def _write_secb_script(self, workspace: 'DockerWorkspace', input_data: ReproducerInput):
        """Write the secb utility script used by agents.

        This script provides commands like 'secb build', 'secb repro', 'secb patch'.

        Args:
            workspace: The Docker workspace
            input_data: Input data
        """
        secb_script = '''#!/bin/bash
# SEC-Bench utility script for building, reproducing, and patching

build() {
    echo "[secb] Building project..."
    cd /src/repo
    if [ -f /src/build.sh ]; then
        bash /src/build.sh
    else
        echo "[secb] ERROR: /src/build.sh not found"
        return 1
    fi
}

repro() {
    echo "[secb] Running exploit to reproduce vulnerability..."
    # This function will be filled by ExploiterAgent
    echo "[secb] ERROR: repro() function not implemented yet"
    return 1
}

patch() {
    echo "[secb] Applying patch..."
    cd /src/repo
    if [ -f /testcase/model_patch.diff ]; then
        git apply /testcase/model_patch.diff
        echo "[secb] Patch applied successfully"
    else
        echo "[secb] ERROR: /testcase/model_patch.diff not found"
        return 1
    fi
}

case "$1" in
    build)
        build
        ;;
    repro)
        repro
        ;;
    patch)
        patch
        ;;
    *)
        echo "Usage: secb {build|repro|patch}"
        exit 1
        ;;
esac
'''
        workspace.execute_command(f"cat > /usr/local/bin/secb << 'EOFSECB'\n{secb_script}\nEOFSECB")
        workspace.execute_command("chmod +x /usr/local/bin/secb")

    async def _execute_phase_with_retry(
        self,
        phase_type: AgentType,
        input_data: ReproducerInput,
        workspace: 'DockerWorkspace',
        previous_output: Optional[PhaseOutput],
    ) -> PhaseOutput:
        """Execute a phase with retry logic.

        Args:
            phase_type: Type of phase to execute
            input_data: Input data for reproduction
            workspace: The Docker workspace (shared across all phases)
            previous_output: Output from previous phase (if any)

        Returns:
            PhaseOutput from execution
        """
        retry_count = 0
        last_output: Optional[PhaseOutput] = None

        while retry_count <= self.max_retries:
            # Create phase input
            phase_input = self._create_phase_input(
                phase_type,
                input_data,
                previous_output,
                retry_count,
            )

            # Execute phase with the shared workspace
            phase_output = await self.phase_executor.execute(
                phase_input,
                self.llm_config,
                workspace,
            )

            # Check if successful
            if phase_output.success:
                return phase_output

            # Store output for potential retry
            last_output = phase_output
            retry_count += 1

            # Log retry
            if retry_count <= self.max_retries:
                logger.warning(
                    f"Phase {phase_type.value} failed, retrying "
                    f"({retry_count}/{self.max_retries})..."
                )

        # All retries exhausted
        if last_output:
            last_output.retry_count = self.max_retries
            return last_output

        # Shouldn't reach here, but return a failed output
        return PhaseOutput(
            phase_type=phase_type,
            success=False,
            final_thought="Phase failed after all retries",
            outputs={},
            artifacts={},
            error_message="Max retries exceeded",
            retry_count=self.max_retries,
        )

    def _create_phase_input(
        self,
        phase_type: AgentType,
        input_data: ReproducerInput,
        previous_output: Optional[PhaseOutput],
        retry_count: int,
    ) -> PhaseInput:
        """Create input for a phase execution.

        Args:
            phase_type: Type of phase
            input_data: Overall input data
            previous_output: Output from previous phase
            retry_count: Current retry count

        Returns:
            PhaseInput for execution
        """
        # Build task description from metadata
        task_description = self._build_task_description(
            phase_type,
            input_data,
            previous_output,
        )

        return PhaseInput(
            phase_type=phase_type,
            task_description=task_description,
            workspace=input_data.workspace,
            instance_id=input_data.instance_id,
            metadata={
                **input_data.metadata,
                'retry_count': retry_count,
                'previous_output': previous_output,
            },
            previous_phase_output=previous_output,
        )

    def _build_task_description(
        self,
        phase_type: AgentType,
        input_data: ReproducerInput,
        previous_output: Optional[PhaseOutput],
    ) -> str:
        """Build task description for a phase.

        Args:
            phase_type: Type of phase
            input_data: Overall input data
            previous_output: Output from previous phase

        Returns:
            Task description string
        """
        # This would normally use Jinja2 templates
        # For now, return a simple description
        base_desc = f"""
Instance ID: {input_data.instance_id}
Repository: {input_data.repository_url}
Base Commit: {input_data.base_commit}
Vulnerability: {input_data.vulnerability_description}

Complete the {phase_type.value} phase.
"""
        return base_desc.strip()
