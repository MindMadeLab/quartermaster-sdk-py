"""Tests for the QuartermasterProvider and auto-configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quartermaster_providers.providers.quartermaster import (
    QuartermasterProvider,
    QUARTERMASTER_MODELS,
)
from quartermaster_providers.registry import ProviderRegistry


class TestQuartermasterProvider:
    """Test QuartermasterProvider initialization."""

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                QuartermasterProvider()

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-test123"}):
            p = QuartermasterProvider()
            assert p.api_key == "qm-test123"

    def test_api_key_from_param(self):
        p = QuartermasterProvider(api_key="qm-direct")
        assert p.api_key == "qm-direct"

    def test_default_base_url(self):
        p = QuartermasterProvider(api_key="qm-test")
        assert "quartermaster" in p.base_url

    def test_custom_base_url(self):
        p = QuartermasterProvider(api_key="qm-test", base_url="http://localhost:9000/v1")
        assert p.base_url == "http://localhost:9000/v1"

    def test_provider_name(self):
        p = QuartermasterProvider(api_key="qm-test")
        assert p.PROVIDER_NAME == "quartermaster"

    def test_cost_returns_none(self):
        p = QuartermasterProvider(api_key="qm-test")
        assert p.get_cost_per_1k_input_tokens("gpt-4o") is None
        assert p.get_cost_per_1k_output_tokens("gpt-4o") is None

    def test_token_estimate_fallback(self):
        p = QuartermasterProvider(api_key="qm-test")
        count = p.estimate_token_count("Hello world, this is a test.", "unknown-model")
        assert count > 0

    def test_known_models_not_empty(self):
        assert len(QUARTERMASTER_MODELS) > 10


class TestAutoConfiguration:
    """Test auto-configuration of QuartermasterProvider in registry."""

    def test_auto_config_with_env_key(self):
        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-auto123"}):
            registry = ProviderRegistry(auto_configure=True)
            assert registry.is_registered("quartermaster")

    def test_no_auto_config_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            registry = ProviderRegistry(auto_configure=True)
            assert not registry.is_registered("quartermaster")

    def test_no_auto_config_when_disabled(self):
        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-test"}):
            registry = ProviderRegistry(auto_configure=False)
            assert not registry.is_registered("quartermaster")

    def test_get_for_model_falls_back_to_quartermaster(self):
        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-fallback"}):
            registry = ProviderRegistry(auto_configure=True)
            # gpt-4o infers "openai" but openai isn't registered — should fallback
            provider = registry.get_for_model("gpt-4o")
            assert provider.PROVIDER_NAME == "quartermaster"

    def test_get_for_model_prefers_explicit_provider(self):
        """If a provider is explicitly registered, use it over Quartermaster."""

        class FakeProvider:
            PROVIDER_NAME = "openai"

            def __init__(self, **kwargs):
                pass

        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-test"}):
            registry = ProviderRegistry(auto_configure=True)
            registry.register("openai", FakeProvider)  # type: ignore[arg-type]
            provider = registry.get_for_model("gpt-4o")
            assert provider.PROVIDER_NAME == "openai"

    def test_get_for_model_unknown_with_quartermaster(self):
        """Unknown model patterns should fall back to Quartermaster if available."""
        with patch.dict("os.environ", {"QUARTERMASTER_API_KEY": "qm-test"}):
            registry = ProviderRegistry(auto_configure=True)
            provider = registry.get_for_model("some-custom-model")
            assert provider.PROVIDER_NAME == "quartermaster"
