"""Domain-specific exceptions."""


class DomainException(Exception):
    """Base exception for domain errors."""

    pass


class InstanceNotFoundError(DomainException):
    """Raised when an instance cannot be found."""

    pass


class ValidationError(DomainException):
    """Raised when validation fails."""

    pass


class PhaseExecutionError(DomainException):
    """Raised when a phase execution fails."""

    def __init__(self, phase: str, message: str):
        self.phase = phase
        super().__init__(f"{phase} phase failed: {message}")


class VerificationError(DomainException):
    """Raised when verification fails."""

    pass
