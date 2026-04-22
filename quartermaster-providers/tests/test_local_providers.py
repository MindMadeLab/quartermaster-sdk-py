"""Tests for local / self-hosted LLM provider classes and registry helpers."""

from __future__ import annotations

import pytest

from quartermaster_providers.providers.local import (
    LOCAL_PROVIDERS,
    LMStudioProvider,
    LlamaCppProvider,
    LocalAIProvider,
    OllamaProvider,
    TGIProvider,
    VLLMProvider,
)
from quartermaster_providers.providers.openai_compat import OpenAICompatibleProvider
from quartermaster_providers.registry import (
    ProviderRegistry,
    infer_provider,
    register_local,
)


# ── Provider class defaults ──────────────────────────────────────────


class TestOllamaProvider:
    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        p = OllamaProvider()
        assert p.base_url == "http://localhost:11434/v1"

    def test_default_no_auth(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_USER", raising=False)
        monkeypatch.delenv("OLLAMA_PASS", raising=False)
        p = OllamaProvider()
        assert p.auth_method == "none"

    def test_provider_name(self):
        p = OllamaProvider()
        assert p.PROVIDER_NAME == "ollama"

    def test_custom_base_url(self):
        p = OllamaProvider(base_url="http://gpu-box:11434/v1")
        assert p.base_url == "http://gpu-box:11434/v1"

    def test_base_url_appends_v1(self):
        p = OllamaProvider(base_url="http://host.docker.internal:11434")
        assert p.base_url == "http://host.docker.internal:11434/v1"

    def test_base_url_preserves_v1(self):
        p = OllamaProvider(base_url="http://host.docker.internal:11434/v1/")
        assert p.base_url == "http://host.docker.internal:11434/v1"

    def test_ollama_host_env_var(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://remote-ollama:11434")
        p = OllamaProvider()
        assert p.base_url == "http://remote-ollama:11434/v1"

    def test_explicit_base_url_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://wrong:11434")
        p = OllamaProvider(base_url="http://right:11434/v1")
        assert p.base_url == "http://right:11434/v1"


class TestVLLMProvider:
    def test_default_base_url(self):
        p = VLLMProvider()
        assert p.base_url == "http://localhost:8000/v1"

    def test_no_auth_by_default(self):
        p = VLLMProvider()
        assert p.auth_method == "none"

    def test_bearer_auth_when_key_given(self):
        p = VLLMProvider(api_key="my-key")
        assert p.auth_method == "bearer"

    def test_provider_name(self):
        p = VLLMProvider()
        assert p.PROVIDER_NAME == "vllm"


class TestLMStudioProvider:
    def test_default_base_url(self):
        p = LMStudioProvider()
        assert p.base_url == "http://localhost:1234/v1"

    def test_provider_name(self):
        p = LMStudioProvider()
        assert p.PROVIDER_NAME == "lm-studio"


class TestTGIProvider:
    def test_default_base_url(self):
        p = TGIProvider()
        assert p.base_url == "http://localhost:8080/v1"

    def test_bearer_with_hf_token(self):
        p = TGIProvider(api_key="hf_abc123")
        assert p.auth_method == "bearer"


class TestLocalAIProvider:
    def test_default_base_url(self):
        p = LocalAIProvider()
        assert p.base_url == "http://localhost:8080/v1"

    def test_provider_name(self):
        p = LocalAIProvider()
        assert p.PROVIDER_NAME == "localai"


class TestLlamaCppProvider:
    def test_default_base_url(self):
        p = LlamaCppProvider()
        assert p.base_url == "http://localhost:8080/v1"

    def test_provider_name(self):
        p = LlamaCppProvider()
        assert p.PROVIDER_NAME == "llama-cpp"


class TestLocalProvidersLookup:
    def test_all_engines_in_lookup(self):
        expected = {"ollama", "vllm", "lm-studio", "tgi", "localai", "llama-cpp"}
        assert set(LOCAL_PROVIDERS.keys()) == expected

    def test_all_are_openai_compatible(self):
        for cls in LOCAL_PROVIDERS.values():
            assert issubclass(cls, OpenAICompatibleProvider)


# ── Registry.register_local() ───────────────────────────────────────


class TestRegisterLocal:
    def test_register_ollama_defaults(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama")
        assert reg.is_registered("ollama")
        provider = reg.get("ollama")
        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://localhost:11434/v1"

    def test_register_vllm_custom_url(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("vllm", base_url="http://gpu:8000/v1")
        provider = reg.get("vllm")
        assert isinstance(provider, VLLMProvider)
        assert provider.base_url == "http://gpu:8000/v1"

    def test_register_with_custom_name(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", name="my-ollama")
        assert reg.is_registered("my-ollama")
        assert not reg.is_registered("ollama")

    def test_register_custom_engine(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local(
            "custom",
            base_url="http://my-server:9000/v1",
            name="my-infra",
        )
        provider = reg.get("my-infra")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.base_url == "http://my-server:9000/v1"

    def test_custom_without_base_url_raises(self):
        reg = ProviderRegistry(auto_configure=False)
        with pytest.raises(Exception, match="requires a base_url"):
            reg.register_local("custom")

    def test_unknown_engine_raises(self):
        reg = ProviderRegistry(auto_configure=False)
        with pytest.raises(Exception, match="Unknown engine"):
            reg.register_local("unknown-engine")

    def test_register_with_model_patterns(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local(
            "ollama",
            models=[r"llama3.*", r"mistral.*", r"codellama.*"],
        )
        # Custom patterns should route to ollama
        provider = reg.get_for_model("llama3.1:70b")
        assert isinstance(provider, OllamaProvider)
        provider = reg.get_for_model("mistral-7b-instruct")
        assert isinstance(provider, OllamaProvider)
        provider = reg.get_for_model("codellama:13b")
        assert isinstance(provider, OllamaProvider)

    def test_custom_patterns_override_builtins(self):
        """Custom patterns checked before built-in MODEL_PATTERNS."""
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", models=[r"^llama-"])
        # Built-in routes llama- to groq, but custom should win
        provider = reg.get_for_model("llama-3.1-70b")
        assert isinstance(provider, OllamaProvider)

    def test_register_as_default_fallback(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", default=True)
        # Unknown model name should fall back to ollama
        provider = reg.get_for_model("my-custom-finetune-v2")
        assert isinstance(provider, OllamaProvider)

    def test_register_vllm_with_api_key(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("vllm", api_key="secret-key")
        provider = reg.get("vllm")
        assert provider.auth_method == "bearer"


# ── add_model_pattern() ─────────────────────────────────────────────


class TestAddModelPattern:
    def test_add_single_pattern(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama")
        reg.add_model_pattern(r"^phi-", "ollama")
        provider = reg.get_for_model("phi-3-mini")
        assert isinstance(provider, OllamaProvider)

    def test_multiple_patterns_for_same_provider(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama")
        reg.add_model_pattern(r"^phi-", "ollama")
        reg.add_model_pattern(r"^qwen", "ollama")
        reg.add_model_pattern(r"^deepseek", "ollama")
        assert isinstance(reg.get_for_model("phi-3-mini"), OllamaProvider)
        assert isinstance(reg.get_for_model("qwen2.5:72b"), OllamaProvider)
        assert isinstance(reg.get_for_model("deepseek-coder-v2"), OllamaProvider)

    def test_clear_removes_custom_patterns(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", models=["^phi-"])
        reg.clear()
        with pytest.raises(Exception):
            reg.get_for_model("phi-3-mini")


# ── set_default_provider() ──────────────────────────────────────────


class TestSetDefaultProvider:
    def test_default_catches_unknown_models(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("vllm")
        reg.set_default_provider("vllm")
        provider = reg.get_for_model("totally-unknown-model")
        assert isinstance(provider, VLLMProvider)

    def test_default_doesnt_override_known_models(self):
        """Built-in patterns still match before the default kicks in."""
        reg = ProviderRegistry(auto_configure=False)
        from quartermaster_providers.providers.openai import OpenAIProvider

        reg.register("openai", OpenAIProvider, api_key="sk-test")
        reg.register_local("ollama", default=True)
        # gpt-4o should still go to openai, not ollama
        provider = reg.get_for_model("gpt-4o")
        assert isinstance(provider, OpenAIProvider)

    def test_clear_removes_default(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", default=True)
        reg.clear()
        with pytest.raises(Exception):
            reg.get_for_model("any-model")


# ── infer_provider with extra_patterns ───────────────────────────────


class TestInferProviderExtraPatterns:
    def test_extra_patterns_checked_first(self):
        extra = [(r"^llama-", "my-ollama")]
        # Built-in maps llama- to groq, but extra should win
        assert infer_provider("llama-3.1", extra_patterns=extra) == "my-ollama"

    def test_falls_back_to_builtin(self):
        extra = [(r"^phi-", "ollama")]
        # gpt- not in extra, should fall through to built-in
        assert infer_provider("gpt-4o", extra_patterns=extra) == "openai"

    def test_none_extra_patterns(self):
        # Should work same as before
        assert infer_provider("claude-sonnet-4-20250514", extra_patterns=None) == "anthropic"

    def test_no_match_returns_none(self):
        extra = [(r"^phi-", "ollama")]
        assert infer_provider("unknown-model", extra_patterns=extra) is None


# ── Integration: mixed local + cloud ─────────────────────────────────


class TestMixedLocalCloud:
    def test_local_and_cloud_coexist(self):
        """Register both cloud and local providers; each resolves correctly."""
        reg = ProviderRegistry(auto_configure=False)
        from quartermaster_providers.providers.openai import OpenAIProvider

        reg.register("openai", OpenAIProvider, api_key="sk-test")
        reg.register_local("ollama", models=[r"llama3.*", r"phi-.*"])

        # Cloud model → OpenAI
        assert isinstance(reg.get_for_model("gpt-4o"), OpenAIProvider)
        # Local model → Ollama
        assert isinstance(reg.get_for_model("llama3.1:70b"), OllamaProvider)
        assert isinstance(reg.get_for_model("phi-3-mini"), OllamaProvider)

    def test_list_providers_includes_local(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama")
        reg.register_local("vllm", name="my-vllm")
        assert "ollama" in reg.list_providers()
        assert "my-vllm" in reg.list_providers()


# ── default_model and module-level register_local() ────────────────────


class TestDefaultModel:
    def test_default_model_routes_literal_name(self):
        """register_local(default_model=...) makes get_for_model find ollama."""
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", default_model="gemma4:26b")
        assert isinstance(reg.get_for_model("gemma4:26b"), OllamaProvider)

    def test_default_model_records_for_get_default_model(self):
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", default_model="gemma4:26b")
        assert reg.get_default_model("ollama") == "gemma4:26b"

    def test_default_model_implies_default_provider(self):
        """Unknown models should fall back to the provider with default_model."""
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", default_model="gemma4:26b")
        # A totally unknown model name still resolves to ollama
        provider = reg.get_for_model("custom-finetune-x")
        assert isinstance(provider, OllamaProvider)
        # And the registry knows what model to use
        assert reg.get_default_model() == "gemma4:26b"

    def test_default_model_with_special_regex_chars(self):
        """Model names containing colon (gemma4:26b) must escape correctly."""
        reg = ProviderRegistry(auto_configure=False)
        reg.register_local("ollama", default_model="llama3.1:70b")
        # The literal name (with colon and dot) must route exactly
        assert isinstance(reg.get_for_model("llama3.1:70b"), OllamaProvider)


class TestModuleLevelRegisterLocal:
    """The user-facing one-liner ``from quartermaster_providers import register_local``."""

    def test_returns_provider_registry(self):
        reg = register_local("ollama")
        assert isinstance(reg, ProviderRegistry)

    def test_user_snippet(self):
        """Mirror the exact 0.1.2 release-note snippet."""
        reg = register_local(
            "ollama",
            base_url="http://host.docker.internal:11434",
            default_model="gemma4:26b",
        )
        provider = reg.get("ollama")
        assert isinstance(provider, OllamaProvider)
        # base_url got the /v1 suffix appended automatically
        assert provider.base_url == "http://host.docker.internal:11434/v1"
        # default_model is recorded for engine-level use
        assert reg.get_default_model("ollama") == "gemma4:26b"
        # Default provider is set so unrouted models resolve here
        assert reg.default_provider == "ollama"

    def test_extending_existing_registry(self):
        """Passing registry= reuses an existing instance instead of building a new one."""
        existing = ProviderRegistry(auto_configure=False)
        returned = register_local("ollama", registry=existing)
        assert returned is existing
        assert existing.is_registered("ollama")
