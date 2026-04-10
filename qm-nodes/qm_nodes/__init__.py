"""qm-nodes: Composable node types for building AI agent graphs."""

from qm_nodes.enums import (
    AvailableErrorHandlingStrategies,
    AvailableExceptionResolutions,
    AvailableMessageTypes,
    AvailableNodeTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)
from qm_nodes.config import AssistantInfo, FlowNodeConf, FlowRunConfig
from qm_nodes.base import AbstractAssistantNode, AbstractLLMAssistantNode
from qm_nodes.chain import Chain, Handler
from qm_nodes.registry import NodeRegistry, register_node

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
