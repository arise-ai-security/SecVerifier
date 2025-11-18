"""Unit tests for LLM Factory."""

import os
from unittest.mock import patch

import pytest

from sec_bench_sdk.infrastructure.sdk.llm_factory import (
    LLMConfig,
    LLMFactory,
    OLLAMA_DEFAULT_BASE_URL,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_config_auto_detects_openai_key(self):
        """Test that OpenAI API key is auto-detected."""
        config = LLMConfig(model="openai/gpt-4o")
        assert config.api_key == "test-openai-key"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_config_auto_detects_anthropic_key(self):
        """Test that Anthropic API key is auto-detected."""
        config = LLMConfig(model="anthropic/claude-sonnet-4-5")
        assert config.api_key == "test-anthropic-key"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"})
    def test_config_auto_detects_google_key(self):
        """Test that Google API key is auto-detected."""
        config = LLMConfig(model="google/gemini-2.0")
        assert config.api_key == "test-google-key"

    def test_config_with_explicit_key(self):
        """Test config with explicitly provided API key."""
        config = LLMConfig(model="openai/gpt-4o", api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_config_default_temperature(self):
        """Test default temperature is 0.0."""
        config = LLMConfig(model="openai/gpt-4o")
        assert config.temperature == 0.0

    def test_config_custom_temperature(self):
        """Test custom temperature."""
        config = LLMConfig(model="openai/gpt-4o", temperature=0.7)
        assert config.temperature == 0.7

    def test_enable_completion_logging_sets_folder(self, tmp_path):
        """Enabling completion logging should record folder path."""
        config = LLMConfig(model="openai/gpt-4o")
        target = tmp_path / "completions"
        config.enable_completion_logging(target)

        assert config.log_completions is True
        assert config.log_completions_folder == str(target)

    @patch.dict(os.environ, {"LLM_API_KEY": "ollama-key"}, clear=True)
    def test_gpt_oss_alias_normalized(self):
        """Ensure gpt-oss aliases automatically map to Ollama."""
        config = LLMConfig(model="gpt-oss:120b-cloud")

        assert config.model == "ollama/gpt-oss:120b-cloud"
        assert config.api_key == "ollama-key"
        assert config.base_url == OLLAMA_DEFAULT_BASE_URL
        assert config.custom_llm_provider == "ollama_chat"

    @patch.dict(
        os.environ,
        {
            "LLM_API_KEY": "ollama-key",
            "OLLAMA_BASE_URL": "https://ollama.example/v1",
        },
        clear=True,
    )
    def test_ollama_respects_custom_base(self):
        """Custom base URLs override defaults for Ollama."""
        config = LLMConfig(model="ollama/gpt-oss:20b-cloud")

        assert config.base_url == "https://ollama.example/v1"
        assert config.ollama_base_url == "https://ollama.example/v1"


class TestLLMFactory:
    """Tests for LLMFactory."""

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_llm_from_config(self, mock_llm):
        """Test creating LLM from config."""
        config = LLMConfig(
            model="openai/gpt-4o",
            temperature=0.5,
            max_output_tokens=4096,
        )

        llm = LLMFactory.create(config)

        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] is None
        assert kwargs["custom_llm_provider"] is None
        assert kwargs["ollama_base_url"] is None
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_output_tokens"] == 4096
        assert kwargs["log_completions"] is False
        assert "log_completions_folder" not in kwargs

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_from_string(self, mock_llm):
        """Test creating LLM from string."""
        llm = LLMFactory.create_from_string("openai/gpt-5-2025-08-07")

        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]["model"] == "openai/gpt-5-2025-08-07"
        assert call_args[1]["api_key"] == "test-key"
        assert call_args[1]["base_url"] is None
        assert call_args[1]["log_completions"] is False
        assert "log_completions_folder" not in call_args[1]
        assert call_args[1]["log_completions"] is False

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_for_phase(self, mock_llm):
        """Test creating LLM for specific phase."""
        llm = LLMFactory.create_for_phase("builder", "openai/gpt-4o")

        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]["model"] == "openai/gpt-4o"
        assert call_args[1]["log_completions"] is False
        assert "log_completions_folder" not in call_args[1]

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_with_completion_logging(self, mock_llm, tmp_path):
        """Configured completion logging should set folder on LLM."""
        config = LLMConfig(model="openai/gpt-4o")
        target = tmp_path / "events"
        config.enable_completion_logging(target)

        LLMFactory.create(config)

        kwargs = mock_llm.call_args.kwargs
        assert kwargs["log_completions"] is True
        assert kwargs["log_completions_folder"] == str(target)

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(
        os.environ,
        {"LLM_API_KEY": "ollama-key", "OLLAMA_BASE_URL": "http://localhost:11434"},
        clear=True,
    )
    def test_create_with_gpt_oss_alias(self, mock_llm):
        """LLMFactory should wire Ollama settings automatically."""
        config = LLMConfig(model="gpt-oss:120b-cloud")
        LLMFactory.create(config)

        mock_llm.assert_called_once()
        kwargs = mock_llm.call_args.kwargs
        assert kwargs["model"] == "ollama/gpt-oss:120b-cloud"
        assert kwargs["api_key"] == "ollama-key"
        assert kwargs["base_url"] == "http://localhost:11434"
        assert kwargs["custom_llm_provider"] == "ollama_chat"
        assert kwargs["ollama_base_url"] == "http://localhost:11434"
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_output_tokens"] == 8192
        assert kwargs["log_completions"] is False
        assert "log_completions_folder" not in kwargs
