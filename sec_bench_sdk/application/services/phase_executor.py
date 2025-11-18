"""Service for executing individual phases using the SDK."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.workspace import DockerWorkspace

from sec_bench_sdk.application.dto.phase_data import PhaseInput, PhaseOutput
from sec_bench_sdk.domain.value_objects import AgentType, PhaseConfig
from sec_bench_sdk.application.services.run_logger import InstanceRunContext
from sec_bench_sdk.infrastructure.sdk.agent_builder import AgentBuilder
from sec_bench_sdk.infrastructure.sdk.conversation_runner import ConversationRunner
from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMFactory, LLMConfig


class PhaseExecutor:
    """Executes individual phases using OpenHands SDK."""

    def __init__(
        self,
        agent_builder: AgentBuilder,
        conversation_runner: ConversationRunner,
    ):
        """Initialize the phase executor.

        Args:
            agent_builder: Builder for creating agents
            conversation_runner: Runner for managing conversations
        """
        self.agent_builder = agent_builder
        self.conversation_runner = conversation_runner

    async def execute(
        self,
        phase_input: PhaseInput,
        llm_config: LLMConfig,
        workspace: DockerWorkspace,
        run_context: InstanceRunContext | None,
    ) -> PhaseOutput:
        """Execute a phase.

        Args:
            phase_input: Input data for the phase
            llm_config: LLM configuration
            workspace: The shared Docker workspace for all phases

        Returns:
            PhaseOutput with results
        """
        # Get phase configuration
        phase_configs = PhaseConfig.get_default_configs()
        phase_config = phase_configs[phase_input.phase_type]

        # Create LLM
        llm = LLMFactory.create(llm_config)

        # Build agent
        agent = self.agent_builder.build_agent(
            llm=llm,
            agent_type=phase_input.phase_type,
            phase_config=phase_config,
            instance_metadata=phase_input.metadata,
        )

        # Get system prompt (for initial message)
        system_prompt = self.agent_builder.get_system_prompt(
            agent_type=phase_input.phase_type,
        )

        # Get task instruction (includes context from previous phases)
        task_instruction = self.agent_builder.get_task_instruction(
            agent_type=phase_input.phase_type,
            instance_metadata=phase_input.metadata,
        )

        # Execute using conversation runner
        output = self.conversation_runner.execute_phase(
            agent=agent,
            system_prompt=system_prompt,
            task_instruction=task_instruction,
            workspace=workspace,
            phase_type=phase_input.phase_type,
            instance_data=phase_input.metadata,
            run_context=run_context,
        )

        return output
