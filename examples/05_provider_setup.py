"""Configure multiple LLM providers with ProviderRegistry.

Shows how to use LLMConfig for request parameters and ProviderRegistry
for managing provider instances.  Also demonstrates automatic provider
inference from model names.
"""

from __future__ import annotations

try:
    from qm_providers.config import LLMConfig
    from qm_providers.base import AbstractLLMProvider
    from qm_providers.registry import ProviderRegistry, infer_provider
except ImportError:
    raise SystemExit("Install qm-providers first:  pip install -e qm-providers")


def main() -> None:
    # -- LLMConfig: unified configuration across providers --------------------
    config = LLMConfig(
        model="gpt-4o",
        provider="openai",
        temperature=0.3,
        max_output_tokens=1024,
        system_message="You are a helpful assistant.",
        stream=False,
    )
    config.validate()  # raises ValueError if invalid
    print("LLMConfig created and validated:")
    print(f"  model={config.model}  provider={config.provider}  temp={config.temperature}")

    # Round-trip through dict
    config_dict = config.to_dict()
    restored = LLMConfig.from_dict(config_dict)
    assert restored.model == config.model
    print("  Round-trip to_dict/from_dict: OK")

    # -- Provider inference from model name -----------------------------------
    print("\nProvider inference:")
    for model in ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash", "grok-3", "llama-3-70b"]:
        provider = infer_provider(model)
        print(f"  {model:30s} -> {provider}")

    # -- ProviderRegistry: managing providers ---------------------------------
    registry = ProviderRegistry()
    print(f"\nEmpty registry providers: {registry.list_providers()}")

    # In a real application you would register actual provider classes:
    #   from qm_providers_openai import OpenAIProvider
    #   registry.register("openai", OpenAIProvider, api_key="sk-...")
    #
    # For this demo we show the registry API without real providers.

    print(f"Is 'openai' registered? {registry.is_registered('openai')}")
    print(f"Is 'anthropic' registered? {registry.is_registered('anthropic')}")

    # -- Validation examples --------------------------------------------------
    print("\nValidation examples:")
    try:
        bad = LLMConfig(model="gpt-4o", provider="openai", temperature=3.0)
        bad.validate()
    except ValueError as e:
        print(f"  Bad temperature: {e}")

    try:
        bad2 = LLMConfig(model="", provider="openai")
        bad2.validate()
    except ValueError as e:
        print(f"  Empty model: {e}")


if __name__ == "__main__":
    main()
