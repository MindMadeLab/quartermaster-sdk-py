"""Tests for MessageRouter."""

from uuid import uuid4

from qm_engine.messaging.message_router import MessageRouter
from qm_engine.stores.memory_store import InMemoryStore
from qm_engine.types import (
    Message,
    MessageRole,
    MessageType,
    ThoughtType,
)
from tests.conftest import make_edge, make_graph, make_node


class TestGetMessagesForNode:
    def setup_method(self):
        self.store = InMemoryStore()
        self.router = MessageRouter(self.store)
        self.flow_id = uuid4()

    def test_skip_thought_returns_empty(self):
        node = make_node(thought_type=ThoughtType.SKIP)
        graph = make_graph([node], [], node)
        msgs = self.router.get_messages_for_node(self.flow_id, node, graph)
        assert msgs == []

    def test_new_thought_with_system_instruction(self):
        node = make_node(
            thought_type=ThoughtType.NEW,
            metadata={"system_instruction": "Be helpful"},
        )
        graph = make_graph([node], [], node)
        msgs = self.router.get_messages_for_node(self.flow_id, node, graph)
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].content == "Be helpful"

    def test_new_thought_without_system_instruction(self):
        node = make_node(thought_type=ThoughtType.NEW)
        graph = make_graph([node], [], node)
        msgs = self.router.get_messages_for_node(self.flow_id, node, graph)
        assert msgs == []

    def test_inherit_gets_last_message_from_predecessors(self):
        pred = make_node(name="pred")
        target = make_node(name="target", thought_type=ThoughtType.INHERIT)
        graph = make_graph([pred, target], [make_edge(pred, target)], pred)

        # Save messages for predecessor
        self.store.save_messages(
            self.flow_id,
            pred.id,
            [
                Message(role=MessageRole.USER, content="first"),
                Message(role=MessageRole.ASSISTANT, content="last"),
            ],
        )

        msgs = self.router.get_messages_for_node(self.flow_id, target, graph)
        assert len(msgs) == 1
        assert msgs[0].content == "last"

    def test_continue_gets_all_messages(self):
        pred = make_node(name="pred")
        target = make_node(name="target", thought_type=ThoughtType.CONTINUE)
        graph = make_graph([pred, target], [make_edge(pred, target)], pred)

        self.store.save_messages(
            self.flow_id,
            pred.id,
            [
                Message(role=MessageRole.USER, content="msg1"),
                Message(role=MessageRole.ASSISTANT, content="msg2"),
            ],
        )

        msgs = self.router.get_messages_for_node(self.flow_id, target, graph)
        assert len(msgs) == 2
        assert msgs[0].content == "msg1"
        assert msgs[1].content == "msg2"


class TestBuildInputMessage:
    def setup_method(self):
        self.store = InMemoryStore()
        self.router = MessageRouter(self.store)

    def test_user_message_type(self):
        node = make_node(message_type=MessageType.USER)
        msg = self.router.build_input_message(node, "hello", {})
        assert msg is not None
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"

    def test_variable_message_type(self):
        node = make_node(
            message_type=MessageType.VARIABLE,
            metadata={"variable_name": "name"},
        )
        msg = self.router.build_input_message(node, "ignored", {"name": "Alice"})
        assert msg is not None
        assert msg.content == "Alice"

    def test_automatic_returns_none(self):
        node = make_node(message_type=MessageType.AUTOMATIC)
        msg = self.router.build_input_message(node, "hello", {})
        assert msg is None
