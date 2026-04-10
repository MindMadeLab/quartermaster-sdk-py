"""Enums for the qm-graph schema, extracted from quartermaster catalog_models."""

from enum import Enum


class NodeType(str, Enum):
    """All available node types in the agent graph."""

    INSTRUCTION = "Instruction1"
    DECISION = "Decision1"
    REASONING = "Reasoning1"
    AGENT = "Agent1"
    START = "Start1"
    END = "End1"
    MERGE = "Merge1"
    IF = "If1"
    SWITCH = "Switch1"
    BREAK = "Break1"
    USER = "User1"
    USER_DECISION = "UserDecision1"
    USER_FORM = "UserForm1"
    STATIC = "Static1"
    VAR = "Var1"
    TEXT = "Text1"
    CODE = "Code1"
    PROGRAM_RUNNER = "ProgramRunner1"
    FLOW_MEMORY = "FlowMemory1"
    READ_MEMORY = "ReadMemory1"
    WRITE_MEMORY = "WriteMemory1"
    TOOL = "Tool1"
    API_CALL = "ApiCall1"
    WEBHOOK = "Webhook1"
    TIMER = "Timer1"
    LOOP = "Loop1"
    PARALLEL = "Parallel1"
    SUB_AGENT = "SubAgent1"
    TEMPLATE = "Template1"
    VALIDATOR = "Validator1"
    TRANSFORMER = "Transformer1"
    FILTER = "Filter1"
    AGGREGATOR = "Aggregator1"
    ROUTER = "Router1"
    ERROR_HANDLER = "ErrorHandler1"
    LOG = "Log1"
    NOTIFICATION = "Notification1"
    CUSTOM = "Custom1"
    COMMENT = "Comment1"


class TraverseIn(str, Enum):
    """How a node handles multiple incoming edges."""

    AWAIT_ALL = "AwaitAll"
    AWAIT_FIRST = "AwaitFirst"


class TraverseOut(str, Enum):
    """How a node dispatches to outgoing edges."""

    SPAWN_ALL = "SpawnAll"
    SPAWN_NONE = "SpawnNone"
    SPAWN_START = "SpawnStart"
    SPAWN_PICKED = "SpawnPickedNode"


class ThoughtType(str, Enum):
    """How a node's execution is displayed in the conversation."""

    SKIP = "SkipThought1"
    NEW = "NewThought1"
    NEW_HIDDEN = "NewHiddenThought1"
    NEW_COLLAPSED = "NewCollapsedThought1"
    EDIT_OR_NEW = "EditSameOrAddNew1"
    EDIT_SAME = "EditSame1"
    APPEND = "AppendThought1"


class MessageType(str, Enum):
    """The role of the message produced by a node."""

    AUTOMATIC = "Automatic"
    USER = "User"
    VARIABLE = "Variable"
    ASSISTANT = "Assistant"
    SYSTEM = "System"


class ErrorStrategy(str, Enum):
    """How errors during node execution are handled."""

    STOP = "Stop"
    RETRY = "Retry"
    SKIP = "Skip"
    CUSTOM = "Custom"
