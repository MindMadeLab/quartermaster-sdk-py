"""Enums defining node types, traversal strategies, thought types, and message types."""

from enum import Enum, auto


class AvailableNodeTypes(Enum):
    """All available node types in the system."""

    Agent1 = "Agent1"
    Blank1 = "Blank1"
    Break1 = "Break1"
    Comment1 = "Comment1"
    End1 = "End1"
    Decision1 = "Decision1"
    Instruction1 = "Instruction1"
    InstructionImageVision1 = "InstructionImageVision1"
    InstructionParameters1 = "InstructionParameters1"
    InstructionProgram1 = "InstructionProgram1"
    InstructionProgramParameters1 = "InstructionProgramParameters1"
    Merge1 = "Merge1"
    Start1 = "Start1"
    Static1 = "Static1"
    StaticMerge1 = "StaticMerge1"
    StaticDecision1 = "StaticDecision1"
    SubAssistant1 = "SubAssistant1"
    ProgramRunner1 = "ProgramRunner1"
    Reasoning1 = "Reasoning1"
    UnselectEnvironment1 = "UnselectEnvironment1"
    UseFile1 = "UseFile1"
    User1 = "User1"
    UserDecision1 = "UserDecision1"
    UserForm1 = "UserForm1"
    UserProgramForm1 = "UserProgramForm1"
    UseEnvironment1 = "UseEnvironment1"
    ViewMetadata1 = "ViewMetadata1"
    StaticProgramParameters1 = "StaticProgramParameters1"
    Code1 = "Code1"
    Var1 = "Var1"
    Text1 = "Text1"
    TextToVariable1 = "TextToVariable1"
    If1 = "If1"
    FlowMemory1 = "FlowMemory1"
    ReadMemory1 = "ReadMemory1"
    WriteMemory1 = "WriteMemory1"
    UpdateMemory1 = "UpdateMemory1"
    UserMemory1 = "UserMemory1"
    Summarize1 = "Summarize1"
    Switch1 = "Switch1"


class AvailableTraversingOut(Enum):
    """How a node spawns outgoing paths."""

    SpawnNone = auto()
    SpawnAll = auto()
    SpawnStart = auto()
    SpawnPickedNode = auto()


class AvailableTraversingIn(Enum):
    """How a node awaits incoming paths."""

    AwaitFirst = auto()
    AwaitAll = auto()


class AvailableThoughtTypes(Enum):
    """How thoughts are created/displayed."""

    SkipThought1 = auto()
    NewThought1 = auto()
    NewHiddenThought1 = auto()
    NewCollapsedThought1 = auto()
    EditSameOrAddNew1 = auto()
    NewHiddenAndNormalThought1 = auto()
    HiddenUserThought1 = auto()
    HiddenAgentThought1 = auto()
    UsePreviousThought1 = auto()


class AvailableMessageTypes(Enum):
    """Message role in conversation."""

    Automatic = "Automatic"
    System = "System"
    User = "User"
    Assistant = "Assistant"
    Tool = "Tool"
    Variable = "Variable"


class AvailableErrorHandlingStrategies(Enum):
    """Error handling strategies for nodes."""

    Stop = "Stop"
    Continue = "Continue"
    Retry = "Retry"


class AvailableExceptionResolutions(Enum):
    """Exception resolution strategies."""

    Retry = auto()
    Break = auto()
    Continue = auto()


# Metadata key constants
NEXT_ASSISTANT_NODE_ID = "next_assistant_node_id"
FUNCTION_SCRIPT_NAME = "function_script_name"
