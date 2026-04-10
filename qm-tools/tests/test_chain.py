"""Tests for qm_tools.chain — Handler and Chain."""

from typing import Any

import pytest

from qm_tools import Chain, Handler


class UpperHandler(Handler):
    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        data["text"] = data["text"].upper()
        return data


class AddPrefixHandler(Handler):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        data["text"] = self.prefix + data["text"]
        return data


class ValidateHandler(Handler):
    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        if "text" not in data:
            raise ValueError("Missing 'text' field")
        return data


class CountHandler(Handler):
    """Handler that tracks how many times it was called."""

    def __init__(self):
        self.call_count = 0

    def handle(self, data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        data["count"] = self.call_count
        return data


class TestHandler:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Handler()

    def test_upper_handler(self):
        h = UpperHandler()
        result = h.handle({"text": "hello"})
        assert result["text"] == "HELLO"

    def test_prefix_handler(self):
        h = AddPrefixHandler(">> ")
        result = h.handle({"text": "hello"})
        assert result["text"] == ">> hello"


class TestChain:
    def test_empty_chain(self):
        chain = Chain()
        result = chain.run({"key": "value"})
        assert result == {"key": "value"}

    def test_single_handler(self):
        chain = Chain().add_handler(UpperHandler())
        result = chain.run({"text": "hello"})
        assert result["text"] == "HELLO"

    def test_multiple_handlers_order(self):
        chain = Chain().add_handler(UpperHandler()).add_handler(AddPrefixHandler(">> "))
        result = chain.run({"text": "hello"})
        assert result["text"] == ">> HELLO"

    def test_reversed_order_different_result(self):
        chain = Chain().add_handler(AddPrefixHandler(">> ")).add_handler(UpperHandler())
        result = chain.run({"text": "hello"})
        assert result["text"] == ">> HELLO"

    def test_handler_raises_stops_chain(self):
        counter = CountHandler()
        chain = Chain().add_handler(ValidateHandler()).add_handler(counter)
        with pytest.raises(ValueError, match="Missing"):
            chain.run({"no_text": True})
        assert counter.call_count == 0

    def test_chain_len(self):
        chain = Chain().add_handler(UpperHandler()).add_handler(ValidateHandler())
        assert len(chain) == 2

    def test_chain_handlers_property(self):
        h1 = UpperHandler()
        h2 = ValidateHandler()
        chain = Chain().add_handler(h1).add_handler(h2)
        handlers = chain.handlers
        assert handlers == [h1, h2]
        # Verify it's a copy
        handlers.append(CountHandler())
        assert len(chain) == 2

    def test_chain_fluent_api(self):
        chain = Chain()
        result = chain.add_handler(UpperHandler())
        assert result is chain  # Returns self

    def test_chain_preserves_extra_keys(self):
        chain = Chain().add_handler(UpperHandler())
        result = chain.run({"text": "hi", "extra": 42})
        assert result["text"] == "HI"
        assert result["extra"] == 42

    def test_chain_data_accumulation(self):
        """Each handler can add new data that subsequent handlers can use."""
        counter = CountHandler()
        chain = (
            Chain()
            .add_handler(UpperHandler())
            .add_handler(counter)
            .add_handler(AddPrefixHandler("result: "))
        )
        result = chain.run({"text": "hello"})
        assert result["text"] == "result: HELLO"
        assert result["count"] == 1

    def test_empty_chain_len(self):
        assert len(Chain()) == 0
