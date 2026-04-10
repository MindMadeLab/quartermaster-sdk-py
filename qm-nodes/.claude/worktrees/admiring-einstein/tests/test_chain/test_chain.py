"""Tests for the Chain class and Handler ABC."""

from typing import Any, Dict

import pytest

from qm_nodes.chain import Chain, Handler


class AddOneHandler(Handler):
    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["value"] = data.get("value", 0) + 1
        return data


class DoubleHandler(Handler):
    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["value"] = data.get("value", 0) * 2
        return data


class AppendHandler(Handler):
    def __init__(self, text: str):
        self.text = text

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["text"] = data.get("text", "") + self.text
        return data


class TestHandler:
    def test_handler_is_abstract(self):
        with pytest.raises(TypeError):
            Handler()  # type: ignore

    def test_handler_subclass_must_implement_handle(self):
        class BadHandler(Handler):
            pass

        with pytest.raises(TypeError):
            BadHandler()  # type: ignore


class TestChain:
    def test_empty_chain_returns_data_unchanged(self):
        chain = Chain()
        data = {"key": "value"}
        result = chain.run(data)
        assert result == {"key": "value"}

    def test_single_handler(self):
        chain = Chain()
        chain.add_handler(AddOneHandler())
        result = chain.run({"value": 0})
        assert result["value"] == 1

    def test_multiple_handlers_in_sequence(self):
        chain = Chain()
        chain.add_handler(AddOneHandler())
        chain.add_handler(DoubleHandler())
        result = chain.run({"value": 0})
        # 0 + 1 = 1, then 1 * 2 = 2
        assert result["value"] == 2

    def test_handler_order_matters(self):
        chain1 = Chain()
        chain1.add_handler(AddOneHandler())
        chain1.add_handler(DoubleHandler())
        result1 = chain1.run({"value": 5})
        # (5 + 1) * 2 = 12

        chain2 = Chain()
        chain2.add_handler(DoubleHandler())
        chain2.add_handler(AddOneHandler())
        result2 = chain2.run({"value": 5})
        # (5 * 2) + 1 = 11

        assert result1["value"] == 12
        assert result2["value"] == 11

    def test_fluent_interface(self):
        result = (
            Chain()
            .add_handler(AddOneHandler())
            .add_handler(DoubleHandler())
            .add_handler(AddOneHandler())
            .run({"value": 0})
        )
        # 0 + 1 = 1, * 2 = 2, + 1 = 3
        assert result["value"] == 3

    def test_data_passes_through_handlers(self):
        chain = Chain()
        chain.add_handler(AppendHandler("Hello"))
        chain.add_handler(AppendHandler(" "))
        chain.add_handler(AppendHandler("World"))
        result = chain.run({})
        assert result["text"] == "Hello World"

    def test_handlers_can_add_keys(self):
        class AddKeyHandler(Handler):
            def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
                data["new_key"] = "new_value"
                return data

        chain = Chain()
        chain.add_handler(AddKeyHandler())
        result = chain.run({"existing": True})
        assert result["new_key"] == "new_value"
        assert result["existing"] is True

    def test_handler_exception_propagates(self):
        class FailHandler(Handler):
            def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
                raise ValueError("Handler failed")

        chain = Chain()
        chain.add_handler(AddOneHandler())
        chain.add_handler(FailHandler())
        chain.add_handler(DoubleHandler())

        with pytest.raises(ValueError, match="Handler failed"):
            chain.run({"value": 0})

    def test_chain_with_many_handlers(self):
        chain = Chain()
        for _ in range(100):
            chain.add_handler(AddOneHandler())
        result = chain.run({"value": 0})
        assert result["value"] == 100
