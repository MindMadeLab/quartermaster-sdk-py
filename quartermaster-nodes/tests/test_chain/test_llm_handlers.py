"""Tests for LLM chain handlers."""

import pytest

from quartermaster_nodes.chain.handlers.llm_handlers import (
    ValidateMemoryID,
    PrepareMessages,
    ContextManager,
    TransformToProvider,
    ProcessStreamResponse,
    CaptureResponse,
)
from quartermaster_nodes.protocols import ContextManagerConfig, LLMConfig


class TestValidateMemoryID:
    def test_valid_memory_id(self):
        handler = ValidateMemoryID()
        data = {"memory_id": "some-id"}
        result = handler.handle(data)
        assert result["memory_id"] == "some-id"

    def test_missing_memory_id_raises(self):
        handler = ValidateMemoryID()
        with pytest.raises(ValueError, match="memory_id is required"):
            handler.handle({})

    def test_none_memory_id_raises(self):
        handler = ValidateMemoryID()
        with pytest.raises(ValueError, match="memory_id is required"):
            handler.handle({"memory_id": None})


class TestPrepareMessages:
    def test_adds_system_message(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai", system_message="You are helpful.")
        handler = PrepareMessages(client=None, llm_config=config)
        result = handler.handle({"messages": []})

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."

    def test_adds_additional_message(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai", system_message=None)
        handler = PrepareMessages(
            client=None,
            llm_config=config,
            additional_message="Choose a path",
            additional_message_role="user",
        )
        result = handler.handle({"messages": []})

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Choose a path"

    def test_preserves_existing_messages(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai", system_message=None)
        handler = PrepareMessages(client=None, llm_config=config)
        existing = [{"role": "user", "content": "Hello"}]
        result = handler.handle({"messages": existing})

        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"

    def test_system_message_added_first(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai", system_message="System prompt")
        handler = PrepareMessages(client=None, llm_config=config)
        result = handler.handle({"messages": [{"role": "user", "content": "Hi"}]})

        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

    def test_no_system_message_when_none(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai", system_message=None)
        handler = PrepareMessages(client=None, llm_config=config)
        result = handler.handle({"messages": []})
        assert len(result["messages"]) == 0


class TestContextManager:
    def test_no_truncation_when_under_limit(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai")
        ctx_config = ContextManagerConfig(max_messages=10)
        handler = ContextManager(client=None, llm_config=config, context_config=ctx_config)

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        result = handler.handle({"messages": messages})
        assert len(result["messages"]) == 5

    def test_truncates_to_max_messages(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai")
        ctx_config = ContextManagerConfig(max_messages=3)
        handler = ContextManager(client=None, llm_config=config, context_config=ctx_config)

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = handler.handle({"messages": messages})
        assert len(result["messages"]) == 3

    def test_preserves_system_message_during_truncation(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai")
        ctx_config = ContextManagerConfig(max_messages=2)
        handler = ContextManager(client=None, llm_config=config, context_config=ctx_config)

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
        ]
        result = handler.handle({"messages": messages})
        assert result["messages"][0]["role"] == "system"
        assert len(result["messages"]) == 3  # system + 2 most recent

    def test_no_truncation_when_max_messages_zero(self):
        config = LLMConfig(model="gpt-4o-mini", provider="openai")
        ctx_config = ContextManagerConfig(max_messages=0)
        handler = ContextManager(client=None, llm_config=config, context_config=ctx_config)

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = handler.handle({"messages": messages})
        assert len(result["messages"]) == 10


class TestTransformToProvider:
    def test_no_transformer_passes_through(self):
        handler = TransformToProvider(transformer=None)
        messages = [{"role": "user", "content": "hello"}]
        result = handler.handle({"messages": messages})
        assert result["messages"] == messages

    def test_applies_transformer(self):
        def uppercase_transformer(messages):
            return [
                {**m, "content": m["content"].upper()}
                for m in messages
            ]

        handler = TransformToProvider(transformer=uppercase_transformer)
        result = handler.handle({"messages": [{"role": "user", "content": "hello"}]})
        assert result["messages"][0]["content"] == "HELLO"


class TestProcessStreamResponse:
    def test_no_response_passes_through(self):
        handler = ProcessStreamResponse()
        result = handler.handle({"some": "data"})
        assert result["some"] == "data"

    def test_processes_simple_response(self):
        class SimpleResponse:
            text = "Hello world"
            tool_calls = []
            usage = {"tokens": 10}

        handler = ProcessStreamResponse()
        result = handler.handle({"response": SimpleResponse()})
        assert result["processed_response"]["text"] == "Hello world"


class TestCaptureResponse:
    def test_no_response_passes_through(self):
        handler = CaptureResponse()
        result = handler.handle({"data": "value"})
        assert "captured_text" not in result

    def test_captures_text(self):
        class MockResponse:
            text = "captured"
            tool_calls = ["tool1"]
            usage = {"tokens": 5}

        handler = CaptureResponse()
        result = handler.handle({"response": MockResponse()})
        assert result["captured_text"] == "captured"
        assert result["captured_tool_calls"] == ["tool1"]
        assert result["captured_usage"] == {"tokens": 5}
