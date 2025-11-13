"""Integration tests for the complete workflow."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sec_bench_sdk.application.dto.phase_data import PhaseOutput, ReproducerInput
from sec_bench_sdk.application.services.phase_executor import PhaseExecutor
from sec_bench_sdk.application.services.reproducer_orchestrator import ReproducerOrchestrator
from sec_bench_sdk.domain.value_objects import AgentType
from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMConfig


class TestWorkflowIntegration:
    """Integration tests for the complete multi-phase workflow."""

    @pytest.fixture
    def mock_phase_executor(self):
        """Create a mock phase executor."""
        executor = MagicMock(spec=PhaseExecutor)
        return executor

    @pytest.fixture
    def llm_config(self):
        """Create a test LLM config."""
        return LLMConfig(model="openai/gpt-4o", api_key="test-key")

    @pytest.fixture
    def reproducer_input(self, tmp_path):
        """Create test reproducer input."""
        return ReproducerInput(
            instance_id="test.cve-2022-1234",
            repository_url="https://github.com/test/repo",
            base_commit="abc123",
            vulnerability_description="Test vulnerability",
            workspace=tmp_path / "workspace",
            output_dir=tmp_path / "output",
            metadata={
                'instance_id': 'test.cve-2022-1234',
                'repository_url': 'https://github.com/test/repo',
            },
        )

    @pytest.mark.asyncio
    async def test_successful_workflow(
        self, mock_phase_executor, llm_config, reproducer_input
    ):
        """Test successful execution of all phases."""
        # Mock successful phase outputs
        builder_output = PhaseOutput(
            phase_type=AgentType.BUILDER,
            success=True,
            final_thought="Build phase completed",
            outputs={'base_commit_hash': 'abc123'},
            artifacts={},
        )
        exploiter_output = PhaseOutput(
            phase_type=AgentType.EXPLOITER,
            success=True,
            final_thought="Exploit phase completed",
            outputs={'poc_created': True},
            artifacts={},
        )
        fixer_output = PhaseOutput(
            phase_type=AgentType.FIXER,
            success=True,
            final_thought="Fix phase completed",
            outputs={'patch_created': True},
            artifacts={},
        )

        # Configure mock to return successful outputs
        mock_phase_executor.execute = AsyncMock(
            side_effect=[builder_output, exploiter_output, fixer_output]
        )

        # Create orchestrator
        orchestrator = ReproducerOrchestrator(
            phase_executor=mock_phase_executor,
            llm_config=llm_config,
            max_retries=1,
        )

        # Execute workflow
        result = await orchestrator.execute(reproducer_input)

        # Verify all phases were executed
        assert mock_phase_executor.execute.call_count == 3
        assert result.success is True
        assert result.builder_output == builder_output
        assert result.exploiter_output == exploiter_output
        assert result.fixer_output == fixer_output

    @pytest.mark.asyncio
    async def test_workflow_stops_on_builder_failure(
        self, mock_phase_executor, llm_config, reproducer_input
    ):
        """Test that workflow stops if builder phase fails."""
        # Mock failed builder output
        builder_output = PhaseOutput(
            phase_type=AgentType.BUILDER,
            success=False,
            final_thought="Build failed",
            outputs={},
            artifacts={},
            error_message="Build error",
        )

        mock_phase_executor.execute = AsyncMock(return_value=builder_output)

        orchestrator = ReproducerOrchestrator(
            phase_executor=mock_phase_executor,
            llm_config=llm_config,
            max_retries=1,
        )

        result = await orchestrator.execute(reproducer_input)

        # Verify only builder was called
        # Due to retry, it might be called twice
        assert mock_phase_executor.execute.call_count >= 1
        assert result.success is False
        assert result.builder_output is not None
        assert result.exploiter_output is None
        assert result.fixer_output is None

    @pytest.mark.asyncio
    async def test_workflow_with_retry(
        self, mock_phase_executor, llm_config, reproducer_input
    ):
        """Test that failed phases are retried."""
        # Mock first failure, then success
        failed_output = PhaseOutput(
            phase_type=AgentType.BUILDER,
            success=False,
            final_thought="Build failed",
            outputs={},
            artifacts={},
            error_message="Build error",
        )
        success_output = PhaseOutput(
            phase_type=AgentType.BUILDER,
            success=True,
            final_thought="Build succeeded on retry",
            outputs={'base_commit_hash': 'abc123'},
            artifacts={},
        )

        mock_phase_executor.execute = AsyncMock(
            side_effect=[failed_output, success_output]
        )

        orchestrator = ReproducerOrchestrator(
            phase_executor=mock_phase_executor,
            llm_config=llm_config,
            max_retries=1,
        )

        result = await orchestrator.execute(reproducer_input)

        # Verify retry occurred
        assert mock_phase_executor.execute.call_count == 2
        assert result.builder_output.success is True
