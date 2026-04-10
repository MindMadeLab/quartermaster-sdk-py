"""Tests for provider registry and model resolution."""

import pytest

from qm_providers.exceptions import InvalidModelError, ProviderError
from qm_providers.registry import ProviderRegistry, infer_provider, get_default_registry
from qm_providers.testing import MockProvider


class TestInferProvider:
    def test_openai_models(self):
        assert infer_provider("gpt-4o") == "openai"
        assert infer_provider("gpt-4-turbo") == "openai"
        assert infer_provider("gpt-3.5-turbo") == "openai"
        assert infer_provider("o1") == "openai"
        assert infer_provider("o1-mini") == "openai"
        assert infer_provider("o3-mini") == "openai"

    def test_anthropic_models(self):
        assert infer_provider("claude-sonnet-4-20250514") == "anthropic"
        assert infer_provider("claude-3-5-sonnet-20241022") == "anthropic"
        assert infer_provider("claude-3-opus-20240229") == "anthropic"
        assert infer_provider("claude-3-haiku-20240307") == "anthropic"

    def test_google_models(self):
        assert infer_provider("gemini-2.0-flash") == "google"
        assert infer_provider("gemini-1.5-pro") == "google"
        assert infer_provider("gemma-7b") == "google"

    def test_groq_models(self):
        assert infer_provider("llama-3.3-70b-versatile") == "groq"
        assert infer_provider("mixtral-8x7b-32768") == "groq"

    def test_xai_models(self):
        assert infer_provider("grok-2") == "xai"
        assert infer_provider("grok-beta") == "xai"

    def test_unknown_model(self):
        assert infer_provider("some-unknown-model") is None
        assert infer_provider("") is None


class TestProviderRegistry:
    def test_register_and_get(self):
        reg = ProviderRegistry()
        reg.register("mock", MockProvider)
        provider = reg.get("mock")
        assert isinstance(provider, MockProvider)

    def test_register_with_kwargs(self):
        reg = ProviderRegistry()
        reg.register(
            "mock",
            MockProvider,
            responses=[],
            models=["test-model"],
        )
        provider = reg.get("mock")
        assert isinstance(provider, MockProvider)

    def test_register_instance(self):
        reg = ProviderRegistry()
        mock = MockProvider()
        reg.register_instance("my-mock", mock)
        assert reg.get("my-mock") is mock

    def test_get_unknown_raises(self):
        reg = ProviderRegistry()
        with pytest.raises(ProviderError, match="not registered"):
            reg.get("nonexistent")

    def test_get_for_model(self):
        reg = ProviderRegistry()
        reg.register("openai", MockProvider)
        reg.register("anthropic", MockProvider)
        reg.register("google", MockProvider)

        provider = reg.get_for_model("gpt-4o")
        assert isinstance(provider, MockProvider)

        provider2 = reg.get_for_model("claude-sonnet-4-20250514")
        assert isinstance(provider2, MockProvider)

    def test_get_for_model_unknown(self):
        reg = ProviderRegistry()
        with pytest.raises(InvalidModelError):
            reg.get_for_model("unknown-model-xyz")

    def test_get_for_model_not_registered(self):
        reg = ProviderRegistry()
        # infer_provider returns "openai" for gpt-4o but it's not registered
        with pytest.raises(ProviderError, match="not registered"):
            reg.get_for_model("gpt-4o")

    def test_list_providers(self):
        reg = ProviderRegistry()
        reg.register("a", MockProvider)
        reg.register("b", MockProvider)
        assert reg.list_providers() == ["a", "b"]

    def test_is_registered(self):
        reg = ProviderRegistry()
        reg.register("mock", MockProvider)
        assert reg.is_registered("mock") is True
        assert reg.is_registered("other") is False

    def test_unregister(self):
        reg = ProviderRegistry()
        reg.register("mock", MockProvider)
        assert reg.is_registered("mock")
        reg.unregister("mock")
        assert not reg.is_registered("mock")

    def test_clear(self):
        reg = ProviderRegistry()
        reg.register("a", MockProvider)
        reg.register("b", MockProvider)
        reg.clear()
        assert reg.list_providers() == []

    def test_lazy_instantiation(self):
        """Provider should only be created on first get()."""
        reg = ProviderRegistry()
        reg.register("mock", MockProvider)
        assert "mock" not in reg._providers
        reg.get("mock")
        assert "mock" in reg._providers

    def test_caches_instance(self):
        """Same instance should be returned on repeated get()."""
        reg = ProviderRegistry()
        reg.register("mock", MockProvider)
        p1 = reg.get("mock")
        p2 = reg.get("mock")
        assert p1 is p2

    def test_re_register_clears_cache(self):
        reg = ProviderRegistry()
        reg.register("mock", MockProvider, models=["a"])
        p1 = reg.get("mock")
        reg.register("mock", MockProvider, models=["b"])
        p2 = reg.get("mock")
        assert p1 is not p2


class TestDefaultRegistry:
    def test_exists(self):
        reg = get_default_registry()
        assert isinstance(reg, ProviderRegistry)

    def test_singleton(self):
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2
