"""qm-engine: Execution engine for AI agent graphs."""

from qm_engine.context.execution_context import ExecutionContext
from qm_engine.context.node_execution import NodeExecution, NodeStatus
from qm_engine.events import (
    FlowError,
    FlowEvent,
    FlowFinished,
    NodeFinished,
    NodeStarted,
    TokenGenerated,
    UserInputRequired,
)
from qm_engine.memory.flow_memory import FlowMemory
from qm_engine.memory.persistent_memory import InMemoryPersistence, PersistentMemory
from qm_engine.messaging.context_manager import ContextManager
from qm_engine.messaging.message_router import MessageRouter
from qm_engine.runner.flow_runner import FlowRunner
from qm_engine.stores.base import ExecutionStore
from qm_engine.stores.memory_store import InMemoryStore
from qm_engine.types import (
    ErrorStrategy,
    GraphEdge,
    GraphNode,
    Message,
    MessageRole,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)

__all__ = [
    # Context
    "ExecutionContext",
    "NodeExecution",
    "NodeStatus",
    # Types & Enums
    "NodeType",
    "TraverseIn",
    "TraverseOut",
    "ThoughtType",
    "MessageType",
    "ErrorStrategy",
    "GraphNode",
    "GraphEdge",
    "Message",
    "MessageRole",
    # Events
    "FlowEvent",
    "NodeStarted",
    "TokenGenerated",
    "NodeFinished",
    "FlowFinished",
    "UserInputRequired",
    "FlowError",
    # Stores
    "ExecutionStore",
    "InMemoryStore",
    # Runner
    "FlowRunner",
    # Memory
    "FlowMemory",
    "PersistentMemory",
    "InMemoryPersistence",
    # Messaging
    "MessageRouter",
    "ContextManager",
]

__version__ = "0.1.0"
