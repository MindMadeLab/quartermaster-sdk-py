"""Tests for the NodeRegistry."""

import pytest

from quartermaster_nodes.base import AbstractAssistantNode
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf
from quartermaster_nodes.enums import (
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.exceptions import NodeNotFoundError
from quartermaster_nodes.registry import NodeRegistry, register_node


class DummyNode(AbstractAssistantNode):
    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Test node"
        info.metadata = {"key": "value"}
        return info

    @classmethod
    def name(cls) -> str:
        return "DummyNode"

    @classmethod
    def version(cls) -> str:
        return "1.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass


class DummyNodeV2(AbstractAssistantNode):
    @classmethod
    def info(cls) -> AssistantInfo:
        info = AssistantInfo()
        info.version = cls.version()
        info.description = "Test node v2"
        info.metadata = {}
        return info

    @classmethod
    def name(cls) -> str:
        return "DummyNode"

    @classmethod
    def version(cls) -> str:
        return "2.0"

    @classmethod
    def flow_config(cls) -> FlowNodeConf:
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn.AwaitFirst,
            traverse_out=AvailableTraversingOut.SpawnAll,
            thought_type=AvailableThoughtTypes.SkipThought1,
            message_type=AvailableMessageTypes.Variable,
        )

    @classmethod
    def think(cls, ctx) -> None:
        pass


class TestNodeRegistry:
    def test_register_and_get(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        assert registry.get("DummyNode") is DummyNode

    def test_get_with_version(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        assert registry.get("DummyNode", "1.0") is DummyNode

    def test_get_nonexistent_raises(self):
        registry = NodeRegistry()
        with pytest.raises(NodeNotFoundError):
            registry.get("NonExistent")

    def test_get_wrong_version_raises(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        with pytest.raises(NodeNotFoundError):
            registry.get("DummyNode", "99.0")

    def test_has_returns_true(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        assert registry.has("DummyNode")
        assert registry.has("DummyNode", "1.0")

    def test_has_returns_false(self):
        registry = NodeRegistry()
        assert not registry.has("NonExistent")

    def test_multiple_versions(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        registry.register(DummyNodeV2)

        assert registry.get("DummyNode", "1.0") is DummyNode
        assert registry.get("DummyNode", "2.0") is DummyNodeV2
        # Latest version (v2) should be default
        assert registry.get("DummyNode") is DummyNodeV2

    def test_count(self):
        registry = NodeRegistry()
        assert registry.count == 0
        registry.register(DummyNode)
        assert registry.count == 1
        registry.register(DummyNodeV2)
        assert registry.count == 2

    def test_list_nodes(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        nodes = registry.list_nodes()

        assert len(nodes) == 1
        assert nodes[0]["name"] == "DummyNode"
        assert nodes[0]["version"] == "1.0"
        assert nodes[0]["description"] == "Test node"
        assert nodes[0]["deprecated"] is False
        assert "flow_config" in nodes[0]
        assert "metadata_schema" in nodes[0]

    def test_catalog_json(self):
        registry = NodeRegistry()
        registry.register(DummyNode)
        catalog = registry.catalog_json()
        assert len(catalog) == 1
        assert catalog[0]["name"] == "DummyNode"

    def test_register_node_decorator(self):
        registry = NodeRegistry()

        @register_node(registry)
        class DecoratedNode(AbstractAssistantNode):
            @classmethod
            def info(cls) -> AssistantInfo:
                info = AssistantInfo()
                info.version = "1.0"
                info.description = "Decorated"
                info.metadata = {}
                return info

            @classmethod
            def name(cls) -> str:
                return "DecoratedNode"

            @classmethod
            def version(cls) -> str:
                return "1.0"

            @classmethod
            def flow_config(cls) -> FlowNodeConf:
                return FlowNodeConf(
                    traverse_in=AvailableTraversingIn.AwaitFirst,
                    traverse_out=AvailableTraversingOut.SpawnAll,
                    thought_type=AvailableThoughtTypes.SkipThought1,
                    message_type=AvailableMessageTypes.Variable,
                )

            @classmethod
            def think(cls, ctx) -> None:
                pass

        assert registry.has("DecoratedNode")
        assert registry.get("DecoratedNode") is DecoratedNode

    def test_discover_nodes(self):
        registry = NodeRegistry()
        count = registry.discover("quartermaster_nodes.nodes")
        assert count > 0  # Should find all our node implementations

    def test_discover_nonexistent_package(self):
        registry = NodeRegistry()
        count = registry.discover("nonexistent.package")
        assert count == 0
