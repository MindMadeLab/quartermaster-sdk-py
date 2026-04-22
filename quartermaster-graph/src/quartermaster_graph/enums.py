"""Enums for the quartermaster-graph schema, extracted from quartermaster catalog_models."""

from enum import Enum


class NodeType(str, Enum):
    """All available node types in the agent graph."""

    AGENT = "Agent1"
    BACK = "Back1"
    BLANK = "Blank1"
    BREAK = "Break1"
    CODE = "Code1"
    COMMENT = "Comment1"
    DECISION = "Decision1"
    END = "End1"
    FLOW_MEMORY = "FlowMemory1"
    IF = "If1"
    INSTRUCTION = "Instruction1"
    INSTRUCTION_FORM = "InstructionForm1"
    INSTRUCTION_IMAGE_VISION = "InstructionImageVision1"
    INSTRUCTION_PARAMETERS = "InstructionParameters1"
    INSTRUCTION_PROGRAM = "InstructionProgram1"
    INSTRUCTION_PROGRAM_PARAMETERS = "InstructionProgramParameters1"
    MERGE = "Merge1"
    PROGRAM_RUNNER = "ProgramRunner1"
    READ_MEMORY = "ReadMemory1"

    START = "Start1"
    STATIC = "Static1"
    STATIC_DECISION = "StaticDecision1"
    STATIC_MERGE = "StaticMerge1"
    STATIC_PROGRAM_PARAMETERS = "StaticProgramParameters1"
    SUB_ASSISTANT = "SubAssistant1"
    SUMMARIZE = "Summarize1"
    SWITCH = "Switch1"
    TEXT = "Text1"
    TEXT_TO_VARIABLE = "TextToVariable1"
    UNSELECT_ENVIRONMENT = "UnselectEnvironment1"
    UPDATE_MEMORY = "UpdateMemory1"
    USE_ENVIRONMENT = "UseEnvironment1"
    USE_FILE = "UseFile1"
    USER = "User1"
    USER_DECISION = "UserDecision1"
    USER_FORM = "UserForm1"
    USER_MEMORY = "UserMemory1"
    USER_PROGRAM_FORM = "UserProgramForm1"
    VAR = "Var1"
    VIEW_METADATA = "ViewMetadata1"
    WRITE_MEMORY = "WriteMemory1"


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
    INHERIT = "InheritThought1"
    CONTINUE = "ContinueThought1"
    NEW_COLLAPSED = "NewCollapsedThought1"
    EDIT_OR_NEW = "EditSameOrAddNew1"
    EDIT_SAME = "EditSame1"
    APPEND = "AppendThought1"
    NEW_HIDDEN_AND_NORMAL = "NewHiddenAndNormalThought1"
    HIDDEN_USER = "HiddenUserThought1"
    HIDDEN_AGENT = "HiddenAgentThought1"
    USE_PREVIOUS = "UsePreviousThought1"


class MessageType(str, Enum):
    """The role of the message produced by a node."""

    AUTOMATIC = "Automatic"
    USER = "User"
    VARIABLE = "Variable"
    ASSISTANT = "Assistant"
    SYSTEM = "System"
    TOOL = "Tool"


class ErrorStrategy(str, Enum):
    """How errors during node execution are handled."""

    STOP = "Stop"
    RETRY = "Retry"
    SKIP = "Skip"
    CUSTOM = "Custom"
    CONTINUE = "Continue"


class ExceptionResolution(str, Enum):
    """Exception resolution strategies."""

    RETRY = "Retry"
    BREAK = "Break"
    CONTINUE = "Continue"
