"""Factory for creating LLM instances with easy model swapping."""

import os
from typing import Optional

from openhands.sdk import LLM


class LLMConfig:
    """Configuration for LLM creation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
    ):
        self.model = model
        self.api_key = api_key or self._get_api_key_from_env(model)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _get_api_key_from_env(model: str) -> str:
        """Automatically determine API key based on model provider."""
        if model.startswith("openai/") or model.startswith("gpt-"):
            return os.getenv("OPENAI_API_KEY", "")
        elif model.startswith("anthropic/") or model.startswith("claude"):
            return os.getenv("ANTHROPIC_API_KEY", "")
        elif model.startswith("google/") or model.startswith("gemini"):
            return os.getenv("GOOGLE_API_KEY", "")
        return os.getenv("LLM_API_KEY", "")


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
        return LLM(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

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
