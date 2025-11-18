"""Factory for creating LLM instances with easy model swapping."""

import os
from pathlib import Path
from typing import Optional

from openhands.sdk import LLM


OLLAMA_DEFAULT_BASE_URL = "https://ollama.com"


class LLMConfig:
    """Configuration for LLM creation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
    ):
        normalized_model = self._normalize_model_name(model)

        self.model = normalized_model
        self.api_key = api_key or self._get_api_key_from_env(normalized_model)
        self.base_url = base_url or self._get_base_url_from_env(normalized_model)
        self.custom_llm_provider = self._get_custom_provider(normalized_model)
        self.ollama_base_url = (
            self.base_url if self._is_ollama_model(normalized_model) else None
        )
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.log_completions: bool = False
        self.log_completions_folder: Optional[str] = None

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        """Map known aliases (like gpt-oss) to provider-prefixed identifiers."""
        trimmed = model.strip()
        lower = trimmed.lower()

        # Support alternative notation "ollama:model"
        if lower.startswith("ollama:"):
            return f"ollama/{trimmed.split(':', 1)[1]}"

        # Already normalized
        if lower.startswith("ollama/"):
            return trimmed

        if "gpt-oss" in lower:
            suffix = trimmed.split("/", 1)[1] if "/" in trimmed else trimmed
            return f"ollama/{suffix}"

        return trimmed

    @staticmethod
    def _get_api_key_from_env(model: str) -> str:
        """Automatically determine API key based on model provider."""
        lower = model.lower()
        if LLMConfig._is_ollama_model(lower):
            # Ollama Cloud uses its own key, but we fall back to generic LLM key
            return os.getenv("OLLAMA_API_KEY") or os.getenv("LLM_API_KEY", "")
        if lower.startswith("openai/") or (lower.startswith("gpt-") and not lower.startswith("gpt-oss")):
            return os.getenv("OPENAI_API_KEY", "")
        if lower.startswith("anthropic/") or lower.startswith("claude"):
            return os.getenv("ANTHROPIC_API_KEY", "")
        if lower.startswith("google/") or lower.startswith("gemini"):
            return os.getenv("GOOGLE_API_KEY", "")
        return os.getenv("LLM_API_KEY", "")

    @staticmethod
    def _get_base_url_from_env(model: str) -> Optional[str]:
        """Return provider-specific base URLs when supplied."""
        lower = model.lower()
        if LLMConfig._is_ollama_model(lower):
            return (
                os.getenv("OLLAMA_BASE_URL")
                or os.getenv("OLLAMA_API_BASE")
                or os.getenv("OLLAMA_HOST")
                or OLLAMA_DEFAULT_BASE_URL
            )
        if lower.startswith("openai/") or (lower.startswith("gpt-") and not lower.startswith("gpt-oss")):
            return os.getenv("OPENAI_BASE_URL")
        if lower.startswith("anthropic/") or lower.startswith("claude"):
            return os.getenv("ANTHROPIC_BASE_URL")
        if lower.startswith("google/") or lower.startswith("gemini"):
            return os.getenv("GOOGLE_API_BASE_URL")
        return os.getenv("LLM_BASE_URL")

    @staticmethod
    def _get_custom_provider(model: str) -> Optional[str]:
        if LLMConfig._is_ollama_model(model.lower()):
            # LiteLLM's chat wrapper expects this provider for streaming/tool use
            return "ollama_chat"
        return None

    @staticmethod
    def _is_ollama_model(model: str) -> bool:
        lower = model.lower()
        return lower.startswith("ollama/") or lower.startswith("ollama:")

    def enable_completion_logging(self, folder: Path | str) -> None:
        """Enable per-instance completion logging."""

        self.log_completions = True
        self.log_completions_folder = str(folder)


class LLMFactory:
    """Factory for creating LLM instances with consistent configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLM:
        """Create an LLM instance from configuration.

        Args:
            config: LLM configuration

        Returns:
            Configured LLM instance

        Examples:
            # Create GPT-5 model
            config = LLMConfig(model="openai/gpt-5-2025-08-07")
            llm = LLMFactory.create(config)

            # Create Claude model
            config = LLMConfig(model="anthropic/claude-sonnet-4-5-20250929")
            llm = LLMFactory.create(config)
        """
        llm_kwargs = {
            "model": config.model,
            "api_key": config.api_key,
            "base_url": config.base_url,
            "custom_llm_provider": config.custom_llm_provider,
            "ollama_base_url": config.ollama_base_url,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "log_completions": config.log_completions,
        }
        if config.log_completions_folder:
            llm_kwargs["log_completions_folder"] = config.log_completions_folder
        return LLM(**llm_kwargs)

    @staticmethod
    def create_from_string(
        model_name: str,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
    ) -> LLM:
        """Create an LLM from a model name string.

        Args:
            model_name: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4-5")
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens to generate

        Returns:
            Configured LLM instance
        """
        config = LLMConfig(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return LLMFactory.create(config)

    @staticmethod
    def create_for_phase(phase_name: str, base_model: str) -> LLM:
        """Create a phase-specific LLM (allows per-phase model configuration).

        Args:
            phase_name: Name of the phase (builder, exploiter, fixer)
            base_model: Base model to use

        Returns:
            Configured LLM instance
        """
        # Could be extended to support different models per phase
        # For now, uses the same base model
        config = LLMConfig(model=base_model)
        return LLMFactory.create(config)
