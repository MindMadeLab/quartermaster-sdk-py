"""Tests for LLMConfig."""

import pytest

from qm_providers.config import LLMConfig


class TestLLMConfig:
    def test_basic_creation(self):
        config = LLMConfig(model="gpt-4o", provider="openai")
        assert config.model == "gpt-4o"
        assert config.provider == "openai"
        assert config.stream is False
        assert config.temperature == 0.7

    def test_all_fields(self):
        config = LLMConfig(
            model="claude-3-opus",
            provider="anthropic",
            stream=True,
            temperature=0.5,
            system_message="You are helpful.",
            max_input_tokens=4000,
            max_output_tokens=2048,
            max_messages=10,
            vision=True,
            thinking_enabled=True,
            thinking_budget=10000,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        assert config.stream is True
        assert config.thinking_enabled is True
        assert config.thinking_budget == 10000
        assert config.top_p == 0.9
        assert config.top_k == 50

    def test_validate_valid(self):
        config = LLMConfig(model="gpt-4o", provider="openai")
        config.validate()  # Should not raise

    def test_validate_empty_model(self):
        config = LLMConfig(model="", provider="openai")
        with pytest.raises(ValueError, match="model is required"):
            config.validate()

    def test_validate_empty_provider(self):
        config = LLMConfig(model="gpt-4o", provider="")
        with pytest.raises(ValueError, match="provider is required"):
            config.validate()

    def test_validate_temperature_range(self):
        config = LLMConfig(model="gpt-4o", provider="openai", temperature=3.0)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

        config2 = LLMConfig(model="gpt-4o", provider="openai", temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            config2.validate()

    def test_validate_max_output_tokens(self):
        config = LLMConfig(model="gpt-4o", provider="openai", max_output_tokens=0)
        with pytest.raises(ValueError, match="max_output_tokens"):
            config.validate()

    def test_validate_max_input_tokens(self):
        config = LLMConfig(model="gpt-4o", provider="openai", max_input_tokens=-1)
        with pytest.raises(ValueError, match="max_input_tokens"):
            config.validate()

    def test_validate_thinking_budget(self):
        config = LLMConfig(model="gpt-4o", provider="openai", thinking_budget=0)
        with pytest.raises(ValueError, match="thinking_budget"):
            config.validate()

    def test_validate_top_p(self):
        config = LLMConfig(model="gpt-4o", provider="openai", top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()

        config2 = LLMConfig(model="gpt-4o", provider="openai", top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            config2.validate()

    def test_validate_top_k(self):
        config = LLMConfig(model="gpt-4o", provider="openai", top_k=0)
        with pytest.raises(ValueError, match="top_k"):
            config.validate()

    def test_validate_frequency_penalty(self):
        config = LLMConfig(model="gpt-4o", provider="openai", frequency_penalty=-3.0)
        with pytest.raises(ValueError, match="frequency_penalty"):
            config.validate()

    def test_validate_presence_penalty(self):
        config = LLMConfig(model="gpt-4o", provider="openai", presence_penalty=3.0)
        with pytest.raises(ValueError, match="presence_penalty"):
            config.validate()

    def test_from_dict(self):
        d = {"model": "gpt-4o", "provider": "openai", "temperature": 0.5}
        config = LLMConfig.from_dict(d)
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5

    def test_from_dict_missing_required(self):
        with pytest.raises(ValueError, match="Missing required fields"):
            LLMConfig.from_dict({"model": "gpt-4o"})

    def test_to_dict(self):
        config = LLMConfig(model="gpt-4o", provider="openai")
        d = config.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["provider"] == "openai"
        assert d["stream"] is False
        assert "temperature" in d

    def test_roundtrip(self):
        config = LLMConfig(
            model="gpt-4o",
            provider="openai",
            temperature=0.3,
            max_output_tokens=1000,
        )
        d = config.to_dict()
        config2 = LLMConfig.from_dict(d)
        assert config.model == config2.model
        assert config.temperature == config2.temperature
        assert config.max_output_tokens == config2.max_output_tokens
