"""Core domain models for security vulnerability instances."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Instance:
    """Represents a security vulnerability instance."""

    instance_id: str
    repository_url: str
    base_commit: str
    vulnerability_description: str
    cve_id: Optional[str] = None
    sanitizer_type: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id cannot be empty")
        if not self.repository_url:
            raise ValueError("repository_url cannot be empty")
        if not self.base_commit:
            raise ValueError("base_commit cannot be empty")


@dataclass
class ExecutionResult:
    """Results from executing all three phases (Build, Exploit, Fix)."""

    builder: Dict[str, Any]
    exploiter: Dict[str, Any]
    fixer: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'builder': self.builder,
            'exploiter': self.exploiter,
            'fixer': self.fixer,
        }


@dataclass
class InstanceOutput:
    """Complete output from processing an instance."""

    execution: ExecutionResult
    build_sh: str
    secb_sh: str
    artifacts: Dict[str, str]  # filename -> base64 encoded content
    env: Dict[str, str]  # environment variables like CFLAGS, CXXFLAGS
    base_commit_hash: Optional[str] = None
    patch: Optional[str] = None
    repo_changes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution': self.execution.to_dict(),
            'build_sh': self.build_sh,
            'secb_sh': self.secb_sh,
            'artifacts': self.artifacts,
            'env': self.env,
            'base_commit_hash': self.base_commit_hash,
            'patch': self.patch,
            'repo_changes': self.repo_changes,
        }


@dataclass
class ReproOutput:
    """Final reproduction output."""

    instance_id: str
    instruction: str
    instance: Dict[str, Any]
    result: InstanceOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'instruction': self.instruction,
            'instance': self.instance,
            'result': self.result.to_dict(),
        }
