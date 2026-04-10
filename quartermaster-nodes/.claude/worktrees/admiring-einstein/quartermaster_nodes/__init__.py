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
from quartermaster_nodes.registry import NodeRegistry, register_node

__version__ = "0.1.0"

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
    # Registry
    "NodeRegistry",
    "register_node",
]
