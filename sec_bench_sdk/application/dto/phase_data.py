"""Data Transfer Objects for phase communication."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from sec_bench_sdk.domain.value_objects import AgentType


@dataclass
class PhaseInput:
    """Input data for a phase execution."""

    phase_type: AgentType
    task_description: str
    workspace: Path
    instance_id: str
    metadata: Dict[str, Any]
    previous_phase_output: Optional['PhaseOutput'] = None


@dataclass
class PhaseOutput:
    """Output data from a phase execution."""

    phase_type: AgentType
    success: bool
    final_thought: str
    outputs: Dict[str, Any]
    artifacts: Dict[str, str]
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class ReproducerInput:
    """Input for the reproducer orchestrator."""

    instance_id: str
    repository_url: str
    base_commit: str
    vulnerability_description: str
    workspace: Path
    output_dir: Path
    metadata: Dict[str, Any]


@dataclass
class ReproducerOutput:
    """Output from the reproducer orchestrator."""

    instance_id: str
    builder_output: Optional[PhaseOutput] = None
    exploiter_output: Optional[PhaseOutput] = None
    fixer_output: Optional[PhaseOutput] = None
    success: bool = False
    error_message: Optional[str] = None
