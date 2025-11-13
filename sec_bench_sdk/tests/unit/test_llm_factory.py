"""Unit tests for LLM Factory."""

import os
from unittest.mock import patch

import pytest

from sec_bench_sdk.infrastructure.sdk.llm_factory import LLMConfig, LLMFactory


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

        mock_llm.assert_called_once_with(
            model="openai/gpt-4o",
            api_key="test-key",
            temperature=0.5,
            max_output_tokens=4096,
        )

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_from_string(self, mock_llm):
        """Test creating LLM from string."""
        llm = LLMFactory.create_from_string("openai/gpt-5-2025-08-07")

        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]["model"] == "openai/gpt-5-2025-08-07"
        assert call_args[1]["api_key"] == "test-key"

    @patch("sec_bench_sdk.infrastructure.sdk.llm_factory.LLM")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_for_phase(self, mock_llm):
        """Test creating LLM for specific phase."""
        llm = LLMFactory.create_for_phase("builder", "openai/gpt-4o")

        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        assert call_args[1]["model"] == "openai/gpt-4o"
