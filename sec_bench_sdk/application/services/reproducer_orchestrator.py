"""Main orchestrator service that coordinates the multi-phase workflow."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import jinja2

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from sec_bench_sdk.application.dto.phase_data import (
    PhaseInput,
    PhaseOutput,
    ReproducerInput,
    ReproducerOutput,
)
from sec_bench_sdk.application.services.phase_executor import PhaseExecutor
from sec_bench_sdk.application.services.run_logger import InstanceRunContext
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

    async def execute(
        self,
        input_data: ReproducerInput,
        run_context: InstanceRunContext | None = None,
    ) -> ReproducerOutput:
        """Execute the complete 3-phase workflow.

        CRITICAL: Uses a SINGLE DockerWorkspace for all phases to maintain state.
        This ensures that files created by BuilderAgent are available to ExploiterAgent,
        and modifications by ExploiterAgent are visible to FixerAgent.

        Args:
            input_data: Input data for reproduction

        Returns:
            ReproducerOutput with results from all phases
        """
        # Import DockerWorkspace with a temporary CWD shim so OpenHands
        # doesn't try to resolve a monorepo root in site-packages
        old_cwd = os.getcwd()
        shim_root = self._ensure_openhands_uv_workspace_shim()
        try:
            os.chdir(shim_root)
            from openhands.workspace import DockerWorkspace  # type: ignore
        finally:
            os.chdir(old_cwd)

        output = ReproducerOutput(instance_id=input_data.instance_id)

        try:
            instance_image = self.docker_manager.get_instance_image(input_data.metadata)

            # Create ONE workspace for all phases - state must persist
            agent_server_image = self.docker_manager.ensure_agent_server_image(
                base_image=instance_image,
                instance_id=input_data.instance_id,
            )
            with DockerWorkspace(
                server_image=agent_server_image,
                host_port=self.host_port,
                extra_ports=self.extra_ports,
                platform=detect_platform(),
            ) as workspace:
                self._setup_workspace(workspace, input_data)

                builder_output = await self._execute_phase_with_retry(
                    AgentType.BUILDER,
                    input_data,
                    workspace,
                    previous_output=None,
                    run_context=run_context,
                )
                output.builder_output = builder_output

                if not builder_output.success:
                    output.success = False
                    output.error_message = f"Builder phase failed: {builder_output.error_message}"
                    return output

                exploiter_output = await self._execute_phase_with_retry(
                    AgentType.EXPLOITER,
                    input_data,
                    workspace,
                    previous_output=builder_output,
                    run_context=run_context,
                )
                output.exploiter_output = exploiter_output

                if not exploiter_output.success:
                    output.success = False
                    output.error_message = f"Exploiter phase failed: {exploiter_output.error_message}"
                    return output

                fixer_output = await self._execute_phase_with_retry(
                    AgentType.FIXER,
                    input_data,
                    workspace,
                    previous_output=exploiter_output,
                    run_context=run_context,
                )
                output.fixer_output = fixer_output

                if not fixer_output.success:
                    output.success = False
                    output.error_message = f"Fixer phase failed: {fixer_output.error_message}"
                    return output

                output.success = True
                return output

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"[{input_data.instance_id}] Error during execution: {error_details}")
            output.success = False
            output.error_message = f"Orchestrator error: {str(e)}\n{error_details}"
            return output

    def _ensure_openhands_uv_workspace_shim(self) -> Path:
        """Create a minimal OpenHands UV workspace layout for import-time checks.

        The OpenHands agent-server build helpers try to discover a UV workspace root
        at import time. When using the pip-installed packages (not the monorepo),
        that discovery fails. We provide a tiny shim that satisfies the discovery:

        - A root `pyproject.toml` with [tool.uv.workspace].members
        - Subfolders: openhands-sdk, openhands-tools, openhands-workspace, openhands-agent-server
          each with a minimal `pyproject.toml`

        Returns:
            Path to the shim root directory.
        """
        # Place the shim inside this repository's workspace dir for stability
        repo_root = Path(__file__).resolve().parents[3]
        shim_root = repo_root / "workspace" / "_openhands_uv_shim"
        shim_root.mkdir(parents=True, exist_ok=True)

        # Root pyproject with UV workspace members
        root_pyproject = shim_root / "pyproject.toml"
        if not root_pyproject.exists():
            root_pyproject.write_text(
                """
[tool.uv.workspace]
members = [
  "openhands-sdk",
  "openhands-tools",
  "openhands-workspace",
  "openhands-agent-server",
]
""".strip()
            )

        # Minimal subproject pyprojects; only `openhands-sdk` might be read for version
        subprojects = [
            ("openhands-sdk", "openhands-sdk", "0.0.0-shim"),
            ("openhands-tools", "openhands-tools", "0.0.0-shim"),
            ("openhands-workspace", "openhands-workspace", "0.0.0-shim"),
            ("openhands-agent-server", "openhands-agent-server", "0.0.0-shim"),
        ]
        for folder, name, version in subprojects:
            subdir = shim_root / folder
            subdir.mkdir(parents=True, exist_ok=True)
            py = subdir / "pyproject.toml"
            if not py.exists():
                py.write_text(
                    f"""
[project]
name = "{name}"
version = "{version}"
""".strip()
                )

        return shim_root

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
            workspace.execute_command("mkdir -p /testcase && chmod 777 /testcase")

            repo_url = input_data.repository_url
            base_commit = input_data.base_commit

            clone_result = workspace.execute_command(
                f"git clone --depth 1 {repo_url} /src/repo || git clone {repo_url} /src/repo"
            )

            # Handle different SDK result schemas
            clone_rc = getattr(clone_result, 'returncode', None)
            if clone_rc is None:
                clone_rc = getattr(clone_result, 'exit_code', None)
            if isinstance(clone_rc, str) and clone_rc.isdigit():
                clone_rc = int(clone_rc)

            if clone_rc not in (0, None):
                logger.warning(f"Clone failed with shallow history, retrying full clone")
                workspace.execute_command(f"git clone {repo_url} /src/repo")

            workspace.execute_command(f"cd /src/repo && git checkout {base_commit}")

            if 'build_sh' in input_data.metadata:
                build_sh_content = input_data.metadata['build_sh']
                workspace.execute_command(
                    f"cat > /src/build.sh << 'EOFBUILDSH'\n{build_sh_content}\nEOFBUILDSH"
                )
                workspace.execute_command("chmod +x /src/build.sh")

            self._write_secb_script(workspace, input_data)

        except Exception as e:
            logger.error(f"Failed to setup workspace: {e}")
            raise

    def _write_secb_script(self, workspace: 'DockerWorkspace', input_data: ReproducerInput):
        """Write the secb utility script used by agents.

        This script provides commands like 'secb build', 'secb repro', 'secb patch'.
        Loads from template to ensure consistency with the original implementation.

        Args:
            workspace: The Docker workspace
            input_data: Input data
        """
        # Load template from infrastructure/scripts
        template_path = Path(__file__).parent.parent.parent / "infrastructure" / "scripts" / "secb_helper.sh.j2"

        if not template_path.exists():
            raise FileNotFoundError(f"secb template not found at {template_path}")

        template = jinja2.Template(template_path.read_text())

        # Render with actual values
        secb_script = template.render(
            instance_id=input_data.instance_id,
            script_name="secb",
            work_dir=input_data.metadata.get('work_dir', '/src/repo')
        )

        workspace.execute_command(f"cat > /usr/local/bin/secb << 'EOFSECB'\n{secb_script}\nEOFSECB")
        workspace.execute_command("chmod +x /usr/local/bin/secb")

    async def _execute_phase_with_retry(
        self,
        phase_type: AgentType,
        input_data: ReproducerInput,
        workspace: 'DockerWorkspace',
        previous_output: Optional[PhaseOutput],
        run_context: InstanceRunContext | None,
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
                run_context,
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
