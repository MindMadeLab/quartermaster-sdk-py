"""qm-graph: Framework-agnostic agent graph schema for AI agent workflows."""

from qm_graph.builder import GraphBuilder
from qm_graph.enums import (
    ErrorStrategy,
    ExceptionResolution,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)
from qm_graph.metadata import (
    CodeMetadata,
    DecisionMetadata,
    IfMetadata,
    InstructionMetadata,
    StaticMetadata,
    SwitchMetadata,
    UserFormMetadata,
    VarMetadata,
    get_metadata_class,
)
from qm_graph.models import (
    Agent,
    AgentVersion,
    GraphDiff,
    GraphEdge,
    GraphNode,
    NodePosition,
)
from qm_graph.serialization import from_json, from_yaml, json_schema, to_json, to_yaml
from qm_graph.templates import Templates
from qm_graph.traversal import (
    find_decision_points,
    find_merge_points,
    get_path,
    get_predecessors,
    get_start_node,
    get_successors,
    topological_sort,
)
from qm_graph.validation import ValidationError, validate_graph
from qm_graph.versioning import bump_major, bump_minor, bump_patch, create_version, fork

__version__ = "0.1.0"

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
    "AgentVersion",
    "GraphDiff",
    "GraphEdge",
    "GraphNode",
    "NodePosition",
    # Metadata
    "CodeMetadata",
    "DecisionMetadata",
    "IfMetadata",
    "InstructionMetadata",
    "StaticMetadata",
    "SwitchMetadata",
    "UserFormMetadata",
    "VarMetadata",
    "get_metadata_class",
    # Validation
    "validate_graph",
    "ValidationError",
    # Versioning
    "bump_major",
    "bump_minor",
    "bump_patch",
    "create_version",
    "fork",
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
    # Templates
    "Templates",
]
