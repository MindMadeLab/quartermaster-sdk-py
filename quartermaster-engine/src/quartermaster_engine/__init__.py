"""quartermaster-engine: Execution engine for AI agent graphs."""

from quartermaster_engine.cancellation import Cancelled
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.context.node_execution import NodeExecution, NodeStatus
from quartermaster_engine.events import (
    CustomEvent,
    FlowError,
    FlowEvent,
    FlowFinished,
    NodeFinished,
    NodeStarted,
    ProgressEvent,
    TokenGenerated,
    ToolCallFinished,
    ToolCallStarted,
    UserInputRequired,
)
from quartermaster_engine.example_runner import (
    AgentExecutor,
    LLMExecutor,
    build_default_registry,
    run_graph,
)
from quartermaster_engine.images import ImageInput, prepare_images
from quartermaster_engine.memory.flow_memory import FlowMemory
from quartermaster_engine.memory.persistent_memory import InMemoryPersistence, PersistentMemory
from quartermaster_engine.messaging.context_manager import ContextManager
from quartermaster_engine.messaging.message_router import MessageRouter
from quartermaster_engine.nodes import (
    NodeExecutor,
    NodeRegistry,
    NodeResult,
    SimpleNodeRegistry,
)
from quartermaster_engine.runner.flow_runner import FlowResult, FlowRunner
from quartermaster_engine.stores.base import ExecutionStore
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.types import (
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
    # Cancellation (v0.4.0)
    "Cancelled",
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
    "ToolCallStarted",
    "ToolCallFinished",
    "ProgressEvent",
    "CustomEvent",
    # Stores
    "ExecutionStore",
    "InMemoryStore",
    # Runner
    "FlowRunner",
    "FlowResult",
    # Node registry / executor protocol
    "NodeRegistry",
    "NodeExecutor",
    "NodeResult",
    "SimpleNodeRegistry",
    # Memory
    "FlowMemory",
    "PersistentMemory",
    "InMemoryPersistence",
    # Messaging
    "MessageRouter",
    "ContextManager",
    # Example runner / default node registry builder
    "run_graph",
    "build_default_registry",
    "LLMExecutor",
    "AgentExecutor",
    # Image-input helpers (v0.3.0 vision kwarg)
    "ImageInput",
    "prepare_images",
]

__version__ = "0.6.2"
