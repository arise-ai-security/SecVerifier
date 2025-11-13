"""Phase-specific result models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PhaseResult:
    """Base class for phase execution results."""

    success: bool
    final_thought: str
    outputs: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class BuildResult(PhaseResult):
    """Result from BuilderAgent execution."""

    base_commit_hash: Optional[str] = None
    repo_changes: Optional[str] = None
    packages_installed: Optional[str] = None


@dataclass
class ExploitResult(PhaseResult):
    """Result from ExploiterAgent execution."""

    poc_created: bool = False
    sanitizer_error_detected: bool = False


@dataclass
class FixResult(PhaseResult):
    """Result from FixerAgent execution."""

    patch_created: bool = False
    patch_applied: bool = False
    verification_passed: bool = False
