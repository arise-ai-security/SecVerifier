"""Unit tests for AgentType value objects."""

from sec_bench_sdk.domain.value_objects.agent_type import (
    AgentType,
    DELEGATION_SEQUENCE,
    PhaseConfig,
)


class TestAgentType:
    """Tests for AgentType enum."""

    def test_agent_types_exist(self):
        """Test that all required agent types exist."""
        assert AgentType.REPRODUCER == "Reproducer"
        assert AgentType.BUILDER == "BuilderAgent"
        assert AgentType.EXPLOITER == "ExploiterAgent"
        assert AgentType.FIXER == "FixerAgent"

    def test_delegation_sequence_order(self):
        """Test that delegation sequence is in correct order."""
        assert len(DELEGATION_SEQUENCE) == 3
        assert DELEGATION_SEQUENCE[0] == AgentType.BUILDER
        assert DELEGATION_SEQUENCE[1] == AgentType.EXPLOITER
        assert DELEGATION_SEQUENCE[2] == AgentType.FIXER


class TestPhaseConfig:
    """Tests for PhaseConfig."""

    def test_phase_config_is_immutable(self):
        """Test that PhaseConfig is immutable."""
        config = PhaseConfig(
            agent_type=AgentType.BUILDER,
            prompt_template="test.j2",
            max_iterations=50,
            timeout_seconds=1800,
        )

        # Should raise error when trying to modify
        try:
            config.max_iterations = 100
            assert False, "Should have raised exception"
        except Exception:
            pass  # Expected

    def test_default_configs_exist(self):
        """Test that default configs exist for all phases."""
        configs = PhaseConfig.get_default_configs()

        assert AgentType.BUILDER in configs
        assert AgentType.EXPLOITER in configs
        assert AgentType.FIXER in configs

    def test_builder_default_config(self):
        """Test builder default configuration."""
        configs = PhaseConfig.get_default_configs()
        builder_config = configs[AgentType.BUILDER]

        assert builder_config.agent_type == AgentType.BUILDER
        assert builder_config.prompt_template == "builder_agent_instruction.j2"
        assert builder_config.max_iterations == 50
        assert builder_config.timeout_seconds == 1800
        assert builder_config.retry_enabled is True
        assert builder_config.max_retries == 1

    def test_exploiter_default_config(self):
        """Test exploiter default configuration."""
        configs = PhaseConfig.get_default_configs()
        exploiter_config = configs[AgentType.EXPLOITER]

        assert exploiter_config.agent_type == AgentType.EXPLOITER
        assert exploiter_config.prompt_template == "exploiter_agent_instruction.j2"

    def test_fixer_default_config(self):
        """Test fixer default configuration."""
        configs = PhaseConfig.get_default_configs()
        fixer_config = configs[AgentType.FIXER]

        assert fixer_config.agent_type == AgentType.FIXER
        assert fixer_config.prompt_template == "fixer_agent_instruction.j2"
