"""Agent type value objects and enumerations."""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass


class AgentType(str, Enum):
    """Types of agents in the system."""

    REPRODUCER = "Reproducer"
    BUILDER = "BuilderAgent"
    EXPLOITER = "ExploiterAgent"
    FIXER = "FixerAgent"


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for a specific phase."""

    agent_type: AgentType
    prompt_template: str
    max_iterations: int
    timeout_seconds: int
    retry_enabled: bool = True
    max_retries: int = 1

    @staticmethod
    def get_default_configs() -> Dict[AgentType, 'PhaseConfig']:
        """Get default configurations for all phases."""
        return {
            AgentType.BUILDER: PhaseConfig(
                agent_type=AgentType.BUILDER,
                prompt_template="builder_agent_instruction.j2",
                max_iterations=50,
                timeout_seconds=1800,  # 30 minutes
                retry_enabled=True,
                max_retries=1,
            ),
            AgentType.EXPLOITER: PhaseConfig(
                agent_type=AgentType.EXPLOITER,
                prompt_template="exploiter_agent_instruction.j2",
                max_iterations=50,
                timeout_seconds=1800,
                retry_enabled=True,
                max_retries=1,
            ),
            AgentType.FIXER: PhaseConfig(
                agent_type=AgentType.FIXER,
                prompt_template="fixer_agent_instruction.j2",
                max_iterations=50,
                timeout_seconds=1800,
                retry_enabled=True,
                max_retries=1,
            ),
        }


# Delegation sequence - order matters!
DELEGATION_SEQUENCE = [AgentType.BUILDER, AgentType.EXPLOITER, AgentType.FIXER]
