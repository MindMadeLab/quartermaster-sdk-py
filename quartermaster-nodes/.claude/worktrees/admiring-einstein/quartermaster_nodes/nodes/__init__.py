"""Node implementations for building AI agent graphs.

Categories:
- llm: Nodes that call LLM providers (Instruction, Decision, Reasoning, Agent, etc.)
- control_flow: Flow control nodes (Start, End, Merge, If, Switch, Break)
- data: Data manipulation nodes (Static, Var, Text, Code, ProgramRunner)
- user_interaction: User-facing nodes (User, UserDecision, UserForm)
- memory: Memory nodes (FlowMemory, ReadMemory, WriteMemory, UpdateMemory, UserMemory)
- utility: Utility nodes (Blank, Comment, ViewMetadata, UseEnvironment, UseFile)
"""

# LLM nodes
from quartermaster_nodes.nodes.llm.instruction import InstructionNodeV1
from quartermaster_nodes.nodes.llm.instruction_image_vision import InstructionImageVision1
from quartermaster_nodes.nodes.llm.instruction_parameters import InstructionParameters1
from quartermaster_nodes.nodes.llm.instruction_program import InstructionProgram1
from quartermaster_nodes.nodes.llm.instruction_program_parameters import InstructionProgramParameters1
from quartermaster_nodes.nodes.llm.decision import Decision1
from quartermaster_nodes.nodes.llm.reasoning import ReasoningV1
from quartermaster_nodes.nodes.llm.agent import AgentNodeV1
from quartermaster_nodes.nodes.llm.summarize import Summarize1
from quartermaster_nodes.nodes.llm.merge import Merge1

# Control flow nodes
from quartermaster_nodes.nodes.control_flow.start import StartNodeV1
from quartermaster_nodes.nodes.control_flow.end import EndNodeV1
from quartermaster_nodes.nodes.control_flow.if_node import IfNode
from quartermaster_nodes.nodes.control_flow.switch import SwitchNode1
from quartermaster_nodes.nodes.control_flow.break_node import BreakNode1
from quartermaster_nodes.nodes.control_flow.sub_assistant import SubAssistant1

# Data nodes
from quartermaster_nodes.nodes.data.static import StaticNode1
from quartermaster_nodes.nodes.data.static_merge import StaticMerge1
from quartermaster_nodes.nodes.data.static_decision import StaticDecision1
from quartermaster_nodes.nodes.data.static_program_parameters import StaticProgramParameters1
from quartermaster_nodes.nodes.data.var import VarNode
from quartermaster_nodes.nodes.data.text import TextNode
from quartermaster_nodes.nodes.data.text_to_variable import TextToVariableNode
from quartermaster_nodes.nodes.data.code import CodeNode
from quartermaster_nodes.nodes.data.program_runner import ProgramRunner1

# User interaction nodes
from quartermaster_nodes.nodes.user_interaction.user import UserNode1
from quartermaster_nodes.nodes.user_interaction.user_decision import UserDecisionV1
from quartermaster_nodes.nodes.user_interaction.user_form import UserFormV1
from quartermaster_nodes.nodes.user_interaction.user_program_form import UserProgramFormV1

# Memory nodes
from quartermaster_nodes.nodes.memory.flow_memory import FlowMemoryNode
from quartermaster_nodes.nodes.memory.read_memory import ReadMemoryNode
from quartermaster_nodes.nodes.memory.write_memory import WriteMemoryNode
from quartermaster_nodes.nodes.memory.update_memory import UpdateMemoryNode
from quartermaster_nodes.nodes.memory.user_memory import UserMemoryNode

# Utility nodes
from quartermaster_nodes.nodes.utility.blank import BlankNode
from quartermaster_nodes.nodes.utility.comment import CommentNode
from quartermaster_nodes.nodes.utility.view_metadata import ViewMetadataNode
from quartermaster_nodes.nodes.utility.use_environment import UseEnvironmentNode
from quartermaster_nodes.nodes.utility.unselect_environment import UnselectEnvironmentNode
from quartermaster_nodes.nodes.utility.use_file import UseFileNode

__all__ = [
    # LLM
    "InstructionNodeV1",
    "InstructionImageVision1",
    "InstructionParameters1",
    "InstructionProgram1",
    "InstructionProgramParameters1",
    "Decision1",
    "ReasoningV1",
    "AgentNodeV1",
    "Summarize1",
    "Merge1",
    # Control flow
    "StartNodeV1",
    "EndNodeV1",
    "IfNode",
    "SwitchNode1",
    "BreakNode1",
    "SubAssistant1",
    # Data
    "StaticNode1",
    "StaticMerge1",
    "StaticDecision1",
    "StaticProgramParameters1",
    "VarNode",
    "TextNode",
    "TextToVariableNode",
    "CodeNode",
    "ProgramRunner1",
    # User interaction
    "UserNode1",
    "UserDecisionV1",
    "UserFormV1",
    "UserProgramFormV1",
    # Memory
    "FlowMemoryNode",
    "ReadMemoryNode",
    "WriteMemoryNode",
    "UpdateMemoryNode",
    "UserMemoryNode",
    # Utility
    "BlankNode",
    "CommentNode",
    "ViewMetadataNode",
    "UseEnvironmentNode",
    "UnselectEnvironmentNode",
    "UseFileNode",
]
