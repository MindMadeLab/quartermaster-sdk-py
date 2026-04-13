"""Tests for enum definitions."""

from quartermaster_graph.enums import (
    ErrorStrategy,
    MessageType,
    NodeType,
    ThoughtType,
    TraverseIn,
    TraverseOut,
)


class TestNodeType:
    def test_member_count(self):
        assert len(NodeType) >= 22

    def test_all_values_end_with_1_or_are_named(self):
        for member in NodeType:
            assert isinstance(member.value, str)
            assert len(member.value) > 0

    def test_essential_types_exist(self):
        essential = [
            NodeType.START, NodeType.END, NodeType.INSTRUCTION,
            NodeType.DECISION, NodeType.IF, NodeType.SWITCH,
            NodeType.MERGE, NodeType.CODE, NodeType.USER,
        ]
        for t in essential:
            assert t in NodeType

    def test_string_enum(self):
        assert NodeType.START == "Start1"
        assert NodeType.INSTRUCTION == "Instruction1"

    def test_serializable(self):
        assert NodeType.START.value == "Start1"
        assert NodeType("Start1") == NodeType.START


class TestTraverseIn:
    def test_values(self):
        assert TraverseIn.AWAIT_ALL == "AwaitAll"
        assert TraverseIn.AWAIT_FIRST == "AwaitFirst"

    def test_count(self):
        assert len(TraverseIn) == 2


class TestTraverseOut:
    def test_values(self):
        assert TraverseOut.SPAWN_ALL == "SpawnAll"
        assert TraverseOut.SPAWN_NONE == "SpawnNone"
        assert TraverseOut.SPAWN_START == "SpawnStart"
        assert TraverseOut.SPAWN_PICKED == "SpawnPickedNode"

    def test_count(self):
        assert len(TraverseOut) == 4


class TestThoughtType:
    def test_has_members(self):
        assert len(ThoughtType) >= 7

    def test_essential_values(self):
        assert ThoughtType.SKIP == "SkipThought1"
        assert ThoughtType.NEW == "NewThought1"


class TestMessageType:
    def test_values(self):
        assert MessageType.AUTOMATIC == "Automatic"
        assert MessageType.USER == "User"
        assert MessageType.ASSISTANT == "Assistant"
        assert MessageType.SYSTEM == "System"
        assert MessageType.VARIABLE == "Variable"


class TestErrorStrategy:
    def test_values(self):
        assert ErrorStrategy.STOP == "Stop"
        assert ErrorStrategy.RETRY == "Retry"
        assert ErrorStrategy.SKIP == "Skip"
        assert ErrorStrategy.CUSTOM == "Custom"

    def test_count(self):
        assert len(ErrorStrategy) >= 4
