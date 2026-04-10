"""Enums defining node types, traversal strategies, thought types, and message types.

All enum types are re-exported from ``quartermaster-graph`` which is the canonical source.
Backward-compatible aliases (``AvailableNodeTypes``, ``AvailableTraversingOut``, etc.)
are provided so that existing call-sites continue to work without changes.
"""

from enum import Enum

from quartermaster_graph.enums import (
    ErrorStrategy,
    ExceptionResolution,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)

# ---------------------------------------------------------------------------
# Backward-compatible aliases
#
# quartermaster-nodes historically used different class names and member-naming
# conventions.  The aliases below map the old names to the canonical
# quartermaster-graph enums so that every ``from quartermaster_nodes.enums import …`` keeps
# working.
# ---------------------------------------------------------------------------

# NodeType  ->  AvailableNodeTypes
# Members are accessed by *value* (e.g. AvailableNodeTypes.Agent1) in old code,
# whereas quartermaster-graph uses UPPER_SNAKE (NodeType.AGENT).  We create a thin subclass
# whose members use the old value-style names.

AvailableNodeTypes = NodeType


class AvailableTraversingOut(Enum):
    """How a node spawns outgoing paths — backward-compatible alias for TraverseOut."""

    SpawnNone = TraverseOut.SPAWN_NONE.value
    SpawnAll = TraverseOut.SPAWN_ALL.value
    SpawnStart = TraverseOut.SPAWN_START.value
    SpawnPickedNode = TraverseOut.SPAWN_PICKED.value


class AvailableTraversingIn(Enum):
    """How a node awaits incoming paths — backward-compatible alias for TraverseIn."""

    AwaitFirst = TraverseIn.AWAIT_FIRST.value
    AwaitAll = TraverseIn.AWAIT_ALL.value


class AvailableThoughtTypes(Enum):
    """How thoughts are created/displayed — backward-compatible alias for ThoughtType."""

    SkipThought1 = ThoughtType.SKIP.value
    NewThought1 = ThoughtType.NEW.value
    NewHiddenThought1 = ThoughtType.NEW_HIDDEN.value
    NewCollapsedThought1 = ThoughtType.NEW_COLLAPSED.value
    EditSameOrAddNew1 = ThoughtType.EDIT_OR_NEW.value
    NewHiddenAndNormalThought1 = ThoughtType.NEW_HIDDEN_AND_NORMAL.value
    HiddenUserThought1 = ThoughtType.HIDDEN_USER.value
    HiddenAgentThought1 = ThoughtType.HIDDEN_AGENT.value
    UsePreviousThought1 = ThoughtType.USE_PREVIOUS.value


class AvailableMessageTypes(Enum):
    """Message role in conversation — backward-compatible alias for MessageType."""

    Automatic = MessageType.AUTOMATIC.value
    System = MessageType.SYSTEM.value
    User = MessageType.USER.value
    Assistant = MessageType.ASSISTANT.value
    Tool = MessageType.TOOL.value
    Variable = MessageType.VARIABLE.value


class AvailableErrorHandlingStrategies(Enum):
    """Error handling strategies — backward-compatible alias for ErrorStrategy."""

    Stop = ErrorStrategy.STOP.value
    Continue = ErrorStrategy.CONTINUE.value
    Retry = ErrorStrategy.RETRY.value


class AvailableExceptionResolutions(Enum):
    """Exception resolution strategies — backward-compatible alias for ExceptionResolution."""

    Retry = ExceptionResolution.RETRY.value
    Break = ExceptionResolution.BREAK.value
    Continue = ExceptionResolution.CONTINUE.value


# Metadata key constants
NEXT_ASSISTANT_NODE_ID = "next_assistant_node_id"
FUNCTION_SCRIPT_NAME = "function_script_name"

__all__ = [
    # Canonical quartermaster-graph enums (re-exported)
    "ErrorStrategy",
    "ExceptionResolution",
    "MessageType",
    "NodeType",
    "ThoughtType",
    "TraverseIn",
    "TraverseOut",
    # Backward-compatible aliases
    "AvailableNodeTypes",
    "AvailableTraversingOut",
    "AvailableTraversingIn",
    "AvailableThoughtTypes",
    "AvailableMessageTypes",
    "AvailableErrorHandlingStrategies",
    "AvailableExceptionResolutions",
    # Constants
    "NEXT_ASSISTANT_NODE_ID",
    "FUNCTION_SCRIPT_NAME",
]
