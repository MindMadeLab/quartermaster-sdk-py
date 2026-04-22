"""quartermaster-nodes: Composable node types for building AI agent graphs."""

from quartermaster_nodes.enums import (
    AvailableErrorHandlingStrategies,
    AvailableExceptionResolutions,
    AvailableMessageTypes,
    AvailableNodeTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from quartermaster_nodes.config import AssistantInfo, FlowNodeConf, FlowRunConfig
from quartermaster_nodes.base import AbstractAssistantNode, AbstractLLMAssistantNode
from quartermaster_nodes.chain import Chain, Handler
from quartermaster_nodes.registry import NodeCatalog, NodeRegistry, register_node

__version__ = "0.4.10"

__all__ = [
    # Enums
    "AvailableErrorHandlingStrategies",
    "AvailableExceptionResolutions",
    "AvailableMessageTypes",
    "AvailableNodeTypes",
    "AvailableThoughtTypes",
    "AvailableTraversingIn",
    "AvailableTraversingOut",
    # Config
    "AssistantInfo",
    "FlowNodeConf",
    "FlowRunConfig",
    # Base classes
    "AbstractAssistantNode",
    "AbstractLLMAssistantNode",
    # Chain
    "Chain",
    "Handler",
    # Catalog (design-time) — NodeRegistry is kept as a deprecated alias
    # for NodeCatalog; see quartermaster_nodes.registry.registry for the
    # reason (naming collision with quartermaster_engine.nodes.NodeRegistry).
    "NodeCatalog",
    "NodeRegistry",
    "register_node",
]
