"""quartermaster-graph: Framework-agnostic agent graph schema for AI agent workflows."""

from quartermaster_graph.builder import GraphBuilder

# Convenience alias -- GraphBuilder IS the graph, so ``Graph`` reads naturally.
Graph = GraphBuilder
from quartermaster_graph.enums import (
    ErrorStrategy,
    ExceptionResolution,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)
from quartermaster_graph.metadata import (
    AgentMetadata,
    BreakMetadata,
    CodeMetadata,
    CommentMetadata,
    DecisionMetadata,
    FlowMemoryMetadata,
    IfMetadata,
    InstructionMetadata,
    LLMMetadata,
    MergeMetadata,
    ReadMemoryMetadata,
    StaticDecisionMetadata,
    StaticMergeMetadata,
    StaticMetadata,
    SubAssistantMetadata,
    SummarizeMetadata,
    SwitchMetadata,
    TextMetadata,
    UserDecisionMetadata,
    UserFormMetadata,
    UserMemoryMetadata,
    UserMetadata,
    VarMetadata,
    WriteMemoryMetadata,
    get_metadata_class,
)
from quartermaster_graph.models import (
    Agent,
    AgentGraph,  # backward-compat alias for GraphSpec
    AgentVersion,  # backward-compat alias for GraphSpec
    GraphEdge,
    GraphNode,
    GraphSpec,
    NodePosition,
)
from quartermaster_graph.serialization import from_json, from_yaml, json_schema, to_json, to_yaml
from quartermaster_graph.templates import Templates
from quartermaster_graph.traversal import (
    find_decision_points,
    find_merge_points,
    get_path,
    get_predecessors,
    get_start_node,
    get_successors,
    topological_sort,
)
from quartermaster_graph.validation import ValidationError, validate_graph

__version__ = "0.2.0"

__all__ = [
    # Enums
    "ErrorStrategy",
    "ExceptionResolution",
    "MessageType",
    "NodeType",
    "ThoughtType",
    "TraverseIn",
    "TraverseOut",
    # Models
    "Agent",
    "GraphSpec",
    "AgentGraph",  # deprecated alias for GraphSpec
    "AgentVersion",  # deprecated alias for GraphSpec
    "GraphEdge",
    "GraphNode",
    "NodePosition",
    # Metadata
    "AgentMetadata",
    "BreakMetadata",
    "CodeMetadata",
    "CommentMetadata",
    "DecisionMetadata",
    "FlowMemoryMetadata",
    "IfMetadata",
    "InstructionMetadata",
    "LLMMetadata",
    "MergeMetadata",
    "ReadMemoryMetadata",
    "StaticDecisionMetadata",
    "StaticMergeMetadata",
    "StaticMetadata",
    "SubAssistantMetadata",
    "SummarizeMetadata",
    "SwitchMetadata",
    "TextMetadata",
    "UserDecisionMetadata",
    "UserFormMetadata",
    "UserMemoryMetadata",
    "UserMetadata",
    "VarMetadata",
    "WriteMemoryMetadata",
    "get_metadata_class",
    # Validation
    "validate_graph",
    "ValidationError",
    # Serialization
    "from_json",
    "from_yaml",
    "to_json",
    "to_yaml",
    "json_schema",
    # Traversal
    "find_decision_points",
    "find_merge_points",
    "get_path",
    "get_predecessors",
    "get_start_node",
    "get_successors",
    "topological_sort",
    # Builder
    "GraphBuilder",
    "Graph",
    # Templates
    "Templates",
]
