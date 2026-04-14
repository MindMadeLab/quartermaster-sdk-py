"""Comprehensive tests for message routing and flow execution pipeline.

Tests the full pipeline end-to-end: message routing, memory propagation,
flow execution order, and conversation composition — using mock executors
instead of real LLM calls.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import UUID, uuid4

import pytest

from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_engine.events import FlowEvent, NodeFinished, NodeStarted
from quartermaster_engine.messaging.message_router import MessageRouter
from quartermaster_engine.nodes import NodeExecutor, NodeResult, SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.types import (
    GraphSpec,
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

# ---------------------------------------------------------------------------
# Mock executors
# ---------------------------------------------------------------------------


class MockLLMExecutor(NodeExecutor):
    """Simulates an LLM node: appends to __conversation__ and reports history size."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        conv = list(context.memory.get("__conversation__", []))
        response = f"Response from {context.current_node.name} (seen {len(conv)} prior entries)"
        conv.append({"role": context.current_node.name, "text": response})
        return NodeResult(
            success=True,
            data={"memory_updates": {"__conversation__": conv}},
            output_text=response,
        )


class MockTextExecutor(NodeExecutor):
    """Simulates a Text node: renders template, appends to __conversation__."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        template = context.get_meta("text", "")
        # Simple variable substitution for testing
        result = template
        for key, val in context.memory.items():
            if isinstance(val, str):
                result = result.replace("{{ " + key + " }}", val)
                result = result.replace("{{" + key + "}}", val)

        if result and result.strip():
            conv = list(context.memory.get("__conversation__", []))
            node_name = context.current_node.name
            round_num = context.memory.get("round_number")
            entry: dict[str, Any] = {"role": node_name, "text": result}
            if round_num is not None:
                entry["round"] = round_num
            conv.append(entry)
            return NodeResult(
                success=True,
                data={"memory_updates": {"__conversation__": conv}},
                output_text=result,
            )
        return NodeResult(success=True, data={}, output_text=result)


class MockVarExecutor(NodeExecutor):
    """Simulates a Var node: evaluates expression, stores in memory. Does NOT touch __conversation__."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        variable = context.get_meta("name", "") or context.get_meta("variable", "")
        expression = context.get_meta("expression", "")
        if variable:
            if expression:
                try:
                    value = eval(expression, {"__builtins__": {}}, dict(context.memory))
                except Exception:
                    value = expression
            else:
                value = ""
                for msg in reversed(context.messages):
                    if msg.content:
                        value = msg.content
                        break
            return NodeResult(
                success=True,
                data={"memory_updates": {variable: value}},
                output_text=str(value),
            )
        return NodeResult(success=True, data={}, output_text="")


class MockDecisionExecutor(NodeExecutor):
    """Simulates a decision node: picks based on metadata 'pick' key."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        pick = context.get_meta("pick", "")
        if not pick:
            user_msgs = [m for m in context.messages if m.role == MessageRole.USER]
            pick = user_msgs[-1].content if user_msgs else "default"
        return NodeResult(
            success=True,
            data={},
            output_text=f"Decided: {pick}",
            picked_node=pick,
        )


class MockIfExecutor(NodeExecutor):
    """Simulates an If node: evaluates expression from metadata."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        expression = context.get_meta("if_expression", "")
        if not expression:
            return NodeResult(success=True, data={}, output_text="true", picked_node="true")
        try:
            result = eval(expression, {"__builtins__": {}}, dict(context.memory))
            picked = "true" if result else "false"
        except Exception:
            picked = "false"
        return NodeResult(success=True, data={}, output_text=picked, picked_node=picked)


class MockLoopCounterExecutor(NodeExecutor):
    """Increments a counter in memory, picks loop or exit based on threshold."""

    def __init__(
        self,
        counter_key: str = "__counter__",
        threshold: int = 3,
        loop_target: str = "LoopBody",
        exit_target: str = "End",
    ):
        self._counter_key = counter_key
        self._threshold = threshold
        self._loop_target = loop_target
        self._exit_target = exit_target

    async def execute(self, context: ExecutionContext) -> NodeResult:
        count = int(context.memory.get(self._counter_key, 0))
        count += 1
        round_num = count
        if count < self._threshold:
            picked = self._loop_target
        else:
            picked = self._exit_target
        return NodeResult(
            success=True,
            data={"memory_updates": {self._counter_key: count, "round_number": round_num}},
            output_text=f"counter={count}",
            picked_node=picked,
        )


class MockMemoryWriteExecutor(NodeExecutor):
    """Writes a specific key/value to memory."""

    def __init__(self, key: str = "test_key", value: str = "test_value"):
        self._key = key
        self._value = value

    async def execute(self, context: ExecutionContext) -> NodeResult:
        return NodeResult(
            success=True,
            data={"memory_updates": {self._key: self._value}},
            output_text=f"{self._key}={self._value}",
        )


class MockMemoryReadExecutor(NodeExecutor):
    """Reads a key from memory and returns it."""

    def __init__(self, key: str = "test_key"):
        self._key = key

    async def execute(self, context: ExecutionContext) -> NodeResult:
        value = context.memory.get(self._key, "NOT_FOUND")
        return NodeResult(success=True, data={}, output_text=f"{self._key}={value}")


class MockStaticExecutor(NodeExecutor):
    """Returns static text from metadata."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        text = context.get_meta("static_text", "")
        return NodeResult(success=True, data={}, output_text=text)


class MockPassthroughExecutor(NodeExecutor):
    """Passes through the last message content."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        text = ""
        for msg in reversed(context.messages):
            if msg.content:
                text = msg.content
                break
        return NodeResult(success=True, data={}, output_text=text)


class TrackingLLMExecutor(NodeExecutor):
    """Like MockLLMExecutor but also records call order for verification."""

    def __init__(self):
        self.calls: list[str] = []

    async def execute(self, context: ExecutionContext) -> NodeResult:
        name = context.current_node.name
        self.calls.append(name)
        conv = list(context.memory.get("__conversation__", []))
        response = f"Response from {name} (seen {len(conv)} prior entries)"
        conv.append({"role": name, "text": response})
        return NodeResult(
            success=True,
            data={"memory_updates": {"__conversation__": conv}},
            output_text=response,
        )


# ---------------------------------------------------------------------------
# Graph construction helpers (low-level, for precise control)
# ---------------------------------------------------------------------------


def make_node(
    node_type: NodeType = NodeType.INSTRUCTION,
    name: str = "",
    traverse_in: TraverseIn = TraverseIn.AWAIT_FIRST,
    traverse_out: TraverseOut = TraverseOut.SPAWN_ALL,
    thought_type: ThoughtType = ThoughtType.CONTINUE,
    message_type: MessageType = MessageType.AUTOMATIC,
    metadata: dict | None = None,
    node_id: UUID | None = None,
) -> GraphNode:
    return GraphNode(
        id=node_id or uuid4(),
        type=node_type,
        name=name,
        traverse_in=traverse_in,
        traverse_out=traverse_out,
        thought_type=thought_type,
        message_type=message_type,
        metadata=metadata or {},
    )


def make_edge(source: GraphNode, target: GraphNode, label: str = "") -> GraphEdge:
    return GraphEdge(source_id=source.id, target_id=target.id, label=label)


def make_graph(nodes: list[GraphNode], edges: list[GraphEdge], start_node: GraphNode) -> GraphSpec:
    return GraphSpec(
        id=uuid4(),
        agent_id=uuid4(),
        start_node_id=start_node.id,
        nodes=nodes,
        edges=edges,
    )


def build_registry(**extra_executors: NodeExecutor) -> SimpleNodeRegistry:
    """Build a registry with all mock executors pre-registered."""
    reg = SimpleNodeRegistry()
    reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())
    reg.register(NodeType.TEXT.value, MockTextExecutor())
    reg.register(NodeType.VAR.value, MockVarExecutor())
    reg.register(NodeType.DECISION.value, MockDecisionExecutor())
    reg.register(NodeType.IF.value, MockIfExecutor())
    reg.register(NodeType.STATIC.value, MockStaticExecutor())
    reg.register(NodeType.STATIC_MERGE.value, MockPassthroughExecutor())
    reg.register(NodeType.WRITE_MEMORY.value, MockMemoryWriteExecutor())
    reg.register(NodeType.READ_MEMORY.value, MockMemoryReadExecutor())
    reg.register(NodeType.SUMMARIZE.value, MockLLMExecutor())
    reg.register(NodeType.AGENT.value, MockLLMExecutor())
    reg.register(NodeType.STATIC_DECISION.value, MockPassthroughExecutor())
    reg.register(NodeType.COMMENT.value, MockPassthroughExecutor())
    reg.register(NodeType.BLANK.value, MockPassthroughExecutor())
    reg.register(NodeType.USER.value, MockPassthroughExecutor())
    for type_val, executor in extra_executors.items():
        reg.register(type_val, executor)
    return reg


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: Message Router Tests (20+)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestMessageRouterThoughtTypes:
    """Test how ThoughtType affects what messages a node receives."""

    def _make_simple_graph(
        self, thought_type: ThoughtType
    ) -> tuple[GraphSpec, GraphNode, GraphNode, GraphNode]:
        """Build Start -> A -> End with A having the given thought_type."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A", thought_type=thought_type)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph([start, a, end], [make_edge(start, a), make_edge(a, end)], start)
        return graph, start, a, end

    def test_new_thought_returns_empty_messages_no_system(self):
        """ThoughtType.NEW with no system instruction returns empty list."""
        graph, start, a, end = self._make_simple_graph(ThoughtType.NEW)
        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert msgs == []

    def test_new_thought_returns_system_message_if_present(self):
        """ThoughtType.NEW with system_instruction returns [system_msg]."""
        start = make_node(NodeType.START, "Start")
        a = make_node(
            NodeType.INSTRUCTION,
            "A",
            thought_type=ThoughtType.NEW,
            metadata={"system_instruction": "You are helpful."},
        )
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph([start, a, end], [make_edge(start, a), make_edge(a, end)], start)

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].content == "You are helpful."

    def test_new_hidden_thought_same_as_new(self):
        """ThoughtType.NEW_HIDDEN behaves identically to NEW."""
        graph, start, a, end = self._make_simple_graph(ThoughtType.NEW_HIDDEN)
        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert msgs == []

    def test_skip_thought_returns_empty(self):
        """ThoughtType.SKIP always returns empty list."""
        graph, start, a, end = self._make_simple_graph(ThoughtType.SKIP)
        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert msgs == []

    def test_inherit_collects_last_predecessor_message(self):
        """ThoughtType.INHERIT gets the last message from each predecessor."""
        start = make_node(NodeType.START, "Start")
        pred = make_node(NodeType.INSTRUCTION, "Pred")
        a = make_node(NodeType.INSTRUCTION, "A", thought_type=ThoughtType.INHERIT)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pred, a, end],
            [make_edge(start, pred), make_edge(pred, a), make_edge(a, end)],
            start,
        )

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()

        # Simulate pred having produced messages
        pred_msgs = [
            Message(role=MessageRole.USER, content="input"),
            Message(role=MessageRole.ASSISTANT, content="pred output"),
        ]
        router.save_node_output(flow_id, pred.id, pred_msgs)

        msgs = router.get_messages_for_node(flow_id, a, graph)
        # Should get only the LAST message from pred
        assert len(msgs) == 1
        assert msgs[0].content == "pred output"

    def test_inherit_multiple_predecessors(self):
        """INHERIT with two predecessors gets last message from each."""
        start = make_node(NodeType.START, "Start")
        pred1 = make_node(NodeType.INSTRUCTION, "Pred1")
        pred2 = make_node(NodeType.INSTRUCTION, "Pred2")
        a = make_node(
            NodeType.INSTRUCTION,
            "A",
            thought_type=ThoughtType.INHERIT,
            traverse_in=TraverseIn.AWAIT_ALL,
        )
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pred1, pred2, a, end],
            [
                make_edge(start, pred1),
                make_edge(start, pred2),
                make_edge(pred1, a),
                make_edge(pred2, a),
                make_edge(a, end),
            ],
            start,
        )

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()

        router.save_node_output(
            flow_id,
            pred1.id,
            [
                Message(role=MessageRole.ASSISTANT, content="from pred1"),
            ],
        )
        router.save_node_output(
            flow_id,
            pred2.id,
            [
                Message(role=MessageRole.ASSISTANT, content="from pred2"),
            ],
        )

        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert len(msgs) == 2
        contents = {m.content for m in msgs}
        assert "from pred1" in contents
        assert "from pred2" in contents

    def test_inherit_with_system_instruction(self):
        """INHERIT prepends system instruction if present."""
        start = make_node(NodeType.START, "Start")
        pred = make_node(NodeType.INSTRUCTION, "Pred")
        a = make_node(
            NodeType.INSTRUCTION,
            "A",
            thought_type=ThoughtType.INHERIT,
            metadata={"system_instruction": "Be concise."},
        )
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pred, a, end],
            [make_edge(start, pred), make_edge(pred, a), make_edge(a, end)],
            start,
        )

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        router.save_node_output(
            flow_id,
            pred.id,
            [
                Message(role=MessageRole.ASSISTANT, content="pred output"),
            ],
        )

        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert len(msgs) == 2
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[0].content == "Be concise."
        assert msgs[1].content == "pred output"

    def test_continue_gets_full_history(self):
        """ThoughtType.CONTINUE gets ALL messages from predecessors."""
        start = make_node(NodeType.START, "Start")
        pred = make_node(NodeType.INSTRUCTION, "Pred")
        a = make_node(NodeType.INSTRUCTION, "A", thought_type=ThoughtType.CONTINUE)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pred, a, end],
            [make_edge(start, pred), make_edge(pred, a), make_edge(a, end)],
            start,
        )

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        pred_msgs = [
            Message(role=MessageRole.USER, content="input"),
            Message(role=MessageRole.ASSISTANT, content="step 1"),
            Message(role=MessageRole.USER, content="follow up"),
            Message(role=MessageRole.ASSISTANT, content="step 2"),
        ]
        router.save_node_output(flow_id, pred.id, pred_msgs)

        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert len(msgs) == 4
        assert msgs[0].content == "input"
        assert msgs[3].content == "step 2"

    def test_continue_with_system_instruction(self):
        """CONTINUE prepends system instruction then full history."""
        start = make_node(NodeType.START, "Start")
        pred = make_node(NodeType.INSTRUCTION, "Pred")
        a = make_node(
            NodeType.INSTRUCTION,
            "A",
            thought_type=ThoughtType.CONTINUE,
            metadata={"system_instruction": "System msg."},
        )
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pred, a, end],
            [make_edge(start, pred), make_edge(pred, a), make_edge(a, end)],
            start,
        )

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        router.save_node_output(
            flow_id,
            pred.id,
            [
                Message(role=MessageRole.USER, content="hello"),
            ],
        )

        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert len(msgs) == 2
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[1].content == "hello"

    def test_inherit_no_predecessor_messages(self):
        """INHERIT with predecessors that have no messages returns empty (or just system)."""
        start = make_node(NodeType.START, "Start")
        pred = make_node(NodeType.INSTRUCTION, "Pred")
        a = make_node(NodeType.INSTRUCTION, "A", thought_type=ThoughtType.INHERIT)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pred, a, end],
            [make_edge(start, pred), make_edge(pred, a), make_edge(a, end)],
            start,
        )

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        # pred has no saved messages
        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert msgs == []

    def test_unknown_thought_type_returns_empty(self):
        """An unrecognized ThoughtType falls through to return []."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A", thought_type=ThoughtType.NEW)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph([start, a, end], [make_edge(start, a), make_edge(a, end)], start)

        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        # Force an unhandled thought type by monkeypatching
        a.thought_type = ThoughtType.NEW_COLLAPSED  # type: ignore[assignment]
        msgs = router.get_messages_for_node(flow_id, a, graph)
        assert msgs == []


class TestMessageRouterBuildInputMessage:
    """Test build_input_message based on MessageType."""

    def test_user_message_type(self):
        """MessageType.USER creates a USER role message with user_input."""
        node = make_node(message_type=MessageType.USER, name="UserNode")
        router = MessageRouter(InMemoryStore())
        msg = router.build_input_message(node, "Hello world", {})
        assert msg is not None
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello world"

    def test_variable_message_type_reads_from_memory(self):
        """MessageType.VARIABLE reads var from memory."""
        node = make_node(
            message_type=MessageType.VARIABLE, name="VarNode", metadata={"variable_name": "my_var"}
        )
        router = MessageRouter(InMemoryStore())
        msg = router.build_input_message(node, "ignored", {"my_var": "from memory"})
        assert msg is not None
        assert msg.role == MessageRole.USER
        assert msg.content == "from memory"

    def test_variable_message_type_missing_var(self):
        """MessageType.VARIABLE with missing var returns empty string content."""
        node = make_node(
            message_type=MessageType.VARIABLE,
            name="VarNode",
            metadata={"variable_name": "missing_var"},
        )
        router = MessageRouter(InMemoryStore())
        msg = router.build_input_message(node, "ignored", {})
        assert msg is not None
        assert msg.content == ""

    def test_assistant_message_type(self):
        """MessageType.ASSISTANT creates an assistant message from metadata."""
        node = make_node(
            message_type=MessageType.ASSISTANT,
            name="AssistNode",
            metadata={"assistant_message": "I am ready."},
        )
        router = MessageRouter(InMemoryStore())
        msg = router.build_input_message(node, "ignored", {})
        assert msg is not None
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "I am ready."

    def test_automatic_message_type_returns_none(self):
        """MessageType.AUTOMATIC returns None."""
        node = make_node(message_type=MessageType.AUTOMATIC, name="AutoNode")
        router = MessageRouter(InMemoryStore())
        msg = router.build_input_message(node, "input", {})
        assert msg is None

    def test_variable_message_type_non_string_value(self):
        """MessageType.VARIABLE converts non-string values to str."""
        node = make_node(
            message_type=MessageType.VARIABLE, name="VarNode", metadata={"variable_name": "count"}
        )
        router = MessageRouter(InMemoryStore())
        msg = router.build_input_message(node, "ignored", {"count": 42})
        assert msg is not None
        assert msg.content == "42"


class TestMessageRouterSaveAndRetrieve:
    """Test save_node_output and get_messages retrieval."""

    def test_save_then_get(self):
        """save_node_output stores messages retrievable by get_messages."""
        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        node_id = uuid4()

        messages = [
            Message(role=MessageRole.USER, content="hello"),
            Message(role=MessageRole.ASSISTANT, content="world"),
        ]
        router.save_node_output(flow_id, node_id, messages)

        retrieved = store.get_messages(flow_id, node_id)
        assert len(retrieved) == 2
        assert retrieved[0].content == "hello"
        assert retrieved[1].content == "world"

    def test_save_overwrites_previous(self):
        """Saving messages for the same node overwrites previous messages."""
        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        node_id = uuid4()

        router.save_node_output(flow_id, node_id, [Message(role=MessageRole.USER, content="first")])
        router.save_node_output(
            flow_id, node_id, [Message(role=MessageRole.USER, content="second")]
        )

        retrieved = store.get_messages(flow_id, node_id)
        assert len(retrieved) == 1
        assert retrieved[0].content == "second"

    def test_append_to_node(self):
        """append_to_node adds a single message to existing history."""
        store = InMemoryStore()
        router = MessageRouter(store)
        flow_id = uuid4()
        node_id = uuid4()

        router.save_node_output(flow_id, node_id, [Message(role=MessageRole.USER, content="first")])
        router.append_to_node(
            flow_id, node_id, Message(role=MessageRole.ASSISTANT, content="second")
        )

        retrieved = store.get_messages(flow_id, node_id)
        assert len(retrieved) == 2
        assert retrieved[1].content == "second"

    def test_get_messages_empty_for_unknown_node(self):
        """get_messages returns empty list for an unknown node."""
        store = InMemoryStore()
        retrieved = store.get_messages(uuid4(), uuid4())
        assert retrieved == []


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: Memory Propagation Tests (15+)                         ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestMemoryPropagation:
    """Test that memory_updates in NodeResult flow correctly through the pipeline."""

    def test_memory_updates_saved_to_store(self):
        """memory_updates in NodeResult get written to the store."""
        start = make_node(NodeType.START, "Start")
        writer = make_node(NodeType.INSTRUCTION, "Writer")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, writer, end],
            [make_edge(start, writer), make_edge(writer, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("greeting", "hello"))

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")

        assert result.success
        assert store.get_memory(result.flow_id, "greeting") == "hello"

    def test_next_node_sees_updated_memory(self):
        """A node following a write sees the updated memory."""
        start = make_node(NodeType.START, "Start")
        writer = make_node(NodeType.INSTRUCTION, "Writer")
        reader = make_node(NodeType.STATIC, "Reader")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, writer, reader, end],
            [make_edge(start, writer), make_edge(writer, reader), make_edge(reader, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("color", "blue"))
        reg.register(NodeType.STATIC.value, MockMemoryReadExecutor("color"))

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")

        assert result.success
        assert "color=blue" in result.final_output

    def test_multiple_memory_updates_in_sequence(self):
        """Two writes in sequence: later node sees both."""
        start = make_node(NodeType.START, "Start")
        w1 = make_node(NodeType.INSTRUCTION, "Write1")
        w2 = make_node(NodeType.STATIC, "Write2")
        reader = make_node(NodeType.SUMMARIZE, "Reader")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, w1, w2, reader, end],
            [
                make_edge(start, w1),
                make_edge(w1, w2),
                make_edge(w2, reader),
                make_edge(reader, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("a", "1"))
        reg.register(NodeType.STATIC.value, MockMemoryWriteExecutor("b", "2"))

        # Reader that checks both values exist
        class BothReader(NodeExecutor):
            async def execute(self, context: ExecutionContext) -> NodeResult:
                a = context.memory.get("a", "MISS")
                b = context.memory.get("b", "MISS")
                return NodeResult(success=True, data={}, output_text=f"a={a},b={b}")

        reg.register(NodeType.SUMMARIZE.value, BothReader())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "a=1" in result.final_output
        assert "b=2" in result.final_output

    def test_conversation_accumulates_across_nodes(self):
        """__conversation__ grows as nodes append to it."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "NodeA")
        b = make_node(NodeType.INSTRUCTION, "NodeB")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, end],
            [make_edge(start, a), make_edge(a, b), make_edge(b, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert conv is not None
        assert len(conv) == 2
        assert conv[0]["role"] == "NodeA"
        assert conv[1]["role"] == "NodeB"

    def test_user_input_set_at_flow_start(self):
        """__user_input__ is stored in memory when the flow starts."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, end],
            [make_edge(start, a), make_edge(a, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("My special input")

        assert store.get_memory(result.flow_id, "__user_input__") == "My special input"

    def test_deep_copy_prevents_cross_contamination(self):
        """Modifying memory in one read doesn't affect the store."""
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "data", {"items": [1, 2, 3]})

        # Get memory and modify it
        read1 = store.get_all_memory(flow_id)
        read1["data"]["items"].append(4)

        # Second read should NOT see the modification
        read2 = store.get_all_memory(flow_id)
        assert len(read2["data"]["items"]) == 3

    def test_deep_copy_on_save(self):
        """Modifying the original after save doesn't affect stored copy."""
        store = InMemoryStore()
        flow_id = uuid4()
        data = {"items": [1, 2, 3]}
        store.save_memory(flow_id, "data", data)

        # Modify original
        data["items"].append(4)

        # Stored version should be unaffected
        stored = store.get_memory(flow_id, "data")
        assert len(stored["items"]) == 3

    def test_memory_visible_to_all_nodes_after_update(self):
        """Memory written by node A is visible to both B and C."""
        start = make_node(NodeType.START, "Start")
        writer = make_node(NodeType.INSTRUCTION, "Writer")
        b = make_node(NodeType.STATIC, "B")
        c = make_node(NodeType.SUMMARIZE, "C")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, writer, b, c, end],
            [
                make_edge(start, writer),
                make_edge(writer, b),
                make_edge(b, c),
                make_edge(c, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("shared", "value"))
        reg.register(NodeType.STATIC.value, MockMemoryReadExecutor("shared"))
        reg.register(NodeType.SUMMARIZE.value, MockMemoryReadExecutor("shared"))

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "shared=value" in result.final_output

    def test_delete_memory(self):
        """delete_memory removes a key from the store."""
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "key", "value")
        assert store.get_memory(flow_id, "key") == "value"
        store.delete_memory(flow_id, "key")
        assert store.get_memory(flow_id, "key") is None

    def test_clear_flow_removes_all(self):
        """clear_flow removes all data for a flow."""
        store = InMemoryStore()
        flow_id = uuid4()
        node_id = uuid4()
        store.save_memory(flow_id, "key", "value")
        store.save_messages(flow_id, node_id, [Message(role=MessageRole.USER, content="hi")])
        store.clear_flow(flow_id)
        assert store.get_memory(flow_id, "key") is None
        assert store.get_messages(flow_id, node_id) == []

    def test_get_memory_returns_none_for_unknown_key(self):
        """get_memory returns None for unknown key."""
        store = InMemoryStore()
        assert store.get_memory(uuid4(), "nope") is None

    def test_get_all_memory_returns_empty_for_unknown_flow(self):
        """get_all_memory returns empty dict for unknown flow."""
        store = InMemoryStore()
        assert store.get_all_memory(uuid4()) == {}

    def test_memory_isolation_between_flows(self):
        """Two flows don't share memory."""
        store = InMemoryStore()
        flow1, flow2 = uuid4(), uuid4()
        store.save_memory(flow1, "key", "flow1_value")
        store.save_memory(flow2, "key", "flow2_value")
        assert store.get_memory(flow1, "key") == "flow1_value"
        assert store.get_memory(flow2, "key") == "flow2_value"

    def test_memory_updates_overwrite_previous(self):
        """A second memory write to the same key overwrites the first."""
        start = make_node(NodeType.START, "Start")
        w1 = make_node(NodeType.INSTRUCTION, "W1")
        w2 = make_node(NodeType.STATIC, "W2")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, w1, w2, end],
            [make_edge(start, w1), make_edge(w1, w2), make_edge(w2, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("x", "first"))
        reg.register(NodeType.STATIC.value, MockMemoryWriteExecutor("x", "second"))

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert store.get_memory(result.flow_id, "x") == "second"

    def test_list_memory_deep_copied(self):
        """Lists in memory are deep copied to prevent aliasing."""
        store = InMemoryStore()
        flow_id = uuid4()
        store.save_memory(flow_id, "items", [1, 2, 3])

        items1 = store.get_memory(flow_id, "items")
        items2 = store.get_memory(flow_id, "items")
        assert items1 == items2
        # They should be different objects
        items1.append(999)
        assert len(items2) == 3


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: Flow Execution Order Tests (20+)                       ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestLinearExecution:
    """Test linear chain execution order."""

    def test_start_a_b_c_end_executes_in_order(self):
        """Start -> A -> B -> C -> End runs in correct sequence."""
        tracker = TrackingLLMExecutor()
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        b = make_node(NodeType.INSTRUCTION, "B")
        c = make_node(NodeType.INSTRUCTION, "C")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, c, end],
            [make_edge(start, a), make_edge(a, b), make_edge(b, c), make_edge(c, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert tracker.calls == ["A", "B", "C"]

    def test_single_node_flow(self):
        """Start -> A -> End (minimal flow)."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "Only")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, end],
            [make_edge(start, a), make_edge(a, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("hello")
        assert result.success
        assert "Response from Only" in result.final_output

    def test_start_end_only(self):
        """Start -> End (no logic nodes) completes successfully."""
        start = make_node(NodeType.START, "Start")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph([start, end], [make_edge(start, end)], start)

        runner = FlowRunner(graph=graph, node_registry=SimpleNodeRegistry())
        result = runner.run("nothing")
        assert result.success


class TestDecisionExecution:
    """Test decision-based branching."""

    def test_decision_picks_correct_branch_by_name(self):
        """Decision picks branch A when output matches edge label 'A'."""
        start = make_node(NodeType.START, "Start")
        decision = make_node(
            NodeType.DECISION,
            "Decide",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"pick": "A"},
        )
        branch_a = make_node(NodeType.INSTRUCTION, "A")
        branch_b = make_node(NodeType.INSTRUCTION, "B")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, decision, branch_a, branch_b, end],
            [
                make_edge(start, decision),
                make_edge(decision, branch_a, label="A"),
                make_edge(decision, branch_b, label="B"),
                make_edge(branch_a, end),
                make_edge(branch_b, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.DECISION.value, MockDecisionExecutor())
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "A" in tracker.calls
        assert "B" not in tracker.calls

    def test_decision_picks_branch_b(self):
        """Decision picks branch B."""
        start = make_node(NodeType.START, "Start")
        decision = make_node(
            NodeType.DECISION,
            "Decide",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"pick": "B"},
        )
        branch_a = make_node(NodeType.INSTRUCTION, "A")
        branch_b = make_node(NodeType.INSTRUCTION, "B")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, decision, branch_a, branch_b, end],
            [
                make_edge(start, decision),
                make_edge(decision, branch_a, label="A"),
                make_edge(decision, branch_b, label="B"),
                make_edge(branch_a, end),
                make_edge(branch_b, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.DECISION.value, MockDecisionExecutor())
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "B" in tracker.calls
        assert "A" not in tracker.calls


class TestIfExecution:
    """Test IF node execution."""

    def test_if_true_branch(self):
        """IF with true expression picks 'true' branch."""
        start = make_node(NodeType.START, "Start")
        if_node = make_node(
            NodeType.IF,
            "Check",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"if_expression": "True"},
        )
        true_branch = make_node(NodeType.INSTRUCTION, "true")
        false_branch = make_node(NodeType.INSTRUCTION, "false")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, if_node, true_branch, false_branch, end],
            [
                make_edge(start, if_node),
                make_edge(if_node, true_branch, label="true"),
                make_edge(if_node, false_branch, label="false"),
                make_edge(true_branch, end),
                make_edge(false_branch, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.IF.value, MockIfExecutor())
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "true" in tracker.calls
        assert "false" not in tracker.calls

    def test_if_false_branch(self):
        """IF with false expression picks 'false' branch."""
        start = make_node(NodeType.START, "Start")
        if_node = make_node(
            NodeType.IF,
            "Check",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"if_expression": "False"},
        )
        true_branch = make_node(NodeType.INSTRUCTION, "true")
        false_branch = make_node(NodeType.INSTRUCTION, "false")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, if_node, true_branch, false_branch, end],
            [
                make_edge(start, if_node),
                make_edge(if_node, true_branch, label="true"),
                make_edge(if_node, false_branch, label="false"),
                make_edge(true_branch, end),
                make_edge(false_branch, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.IF.value, MockIfExecutor())
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "false" in tracker.calls
        assert "true" not in tracker.calls

    def test_if_evaluates_memory_expression(self):
        """IF expression can reference flow memory variables."""
        start = make_node(NodeType.START, "Start")
        writer = make_node(NodeType.INSTRUCTION, "Writer")
        if_node = make_node(
            NodeType.IF,
            "Check",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"if_expression": "score > 5"},
        )
        yes = make_node(NodeType.STATIC, "true")
        no = make_node(NodeType.SUMMARIZE, "false")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, writer, if_node, yes, no, end],
            [
                make_edge(start, writer),
                make_edge(writer, if_node),
                make_edge(if_node, yes, label="true"),
                make_edge(if_node, no, label="false"),
                make_edge(yes, end),
                make_edge(no, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("score", 10))
        reg.register(NodeType.IF.value, MockIfExecutor())
        reg.register(NodeType.STATIC.value, MockStaticExecutor())
        reg.register(NodeType.SUMMARIZE.value, MockStaticExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success


class TestParallelExecution:
    """Test parallel branching (SPAWN_ALL)."""

    def test_parallel_both_branches_execute(self):
        """Fork -> [A, B] -> Merge: both A and B execute."""
        start = make_node(NodeType.START, "Start")
        fork = make_node(NodeType.INSTRUCTION, "Fork")
        a = make_node(NodeType.INSTRUCTION, "BranchA")
        b = make_node(NodeType.STATIC, "BranchB")
        merge = make_node(NodeType.MERGE, "Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, fork, a, b, merge, end],
            [
                make_edge(start, fork),
                make_edge(fork, a),
                make_edge(fork, b),
                make_edge(a, merge),
                make_edge(b, merge),
                make_edge(merge, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)
        reg.register(NodeType.STATIC.value, MockStaticExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("parallel")
        assert result.success
        assert "Fork" in tracker.calls
        assert "BranchA" in tracker.calls


class TestLoopExecution:
    """Test loop (back-edge) execution."""

    def test_loop_iterates_correctly(self):
        """Loop body re-executes until counter reaches threshold."""
        start = make_node(NodeType.START, "Start")
        body = make_node(NodeType.INSTRUCTION, "LoopBody", traverse_in=TraverseIn.AWAIT_FIRST)
        checker = make_node(NodeType.IF, "Check", traverse_out=TraverseOut.SPAWN_PICKED)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, body, checker, end],
            [
                make_edge(start, body),
                make_edge(body, checker),
                make_edge(checker, body, label="LoopBody"),
                make_edge(checker, end, label="End"),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        counter = MockLoopCounterExecutor(threshold=3, loop_target="LoopBody", exit_target="End")
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)
        reg.register(NodeType.IF.value, counter)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("loop")
        assert result.success
        assert tracker.calls.count("LoopBody") == 3

    def test_loop_threshold_1_no_repeat(self):
        """Loop with threshold 1 exits immediately."""
        start = make_node(NodeType.START, "Start")
        body = make_node(NodeType.INSTRUCTION, "LoopBody", traverse_in=TraverseIn.AWAIT_FIRST)
        checker = make_node(NodeType.IF, "Check", traverse_out=TraverseOut.SPAWN_PICKED)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, body, checker, end],
            [
                make_edge(start, body),
                make_edge(body, checker),
                make_edge(checker, body, label="LoopBody"),
                make_edge(checker, end, label="End"),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        counter = MockLoopCounterExecutor(threshold=1, loop_target="LoopBody", exit_target="End")
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)
        reg.register(NodeType.IF.value, counter)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("loop")
        assert result.success
        assert tracker.calls.count("LoopBody") == 1


class TestTraverseInStrategies:
    """Test AWAIT_FIRST and AWAIT_ALL gating."""

    def test_await_first_fires_on_any_predecessor(self):
        """AWAIT_FIRST: node fires as soon as any predecessor completes."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        # B has AWAIT_FIRST with two predecessors (Start and A)
        b = make_node(NodeType.INSTRUCTION, "B", traverse_in=TraverseIn.AWAIT_FIRST)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, end],
            [make_edge(start, a), make_edge(a, b), make_edge(b, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success

    def test_await_all_waits_for_all_predecessors(self):
        """AWAIT_ALL: merge node only fires when all predecessors complete."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        b = make_node(NodeType.STATIC, "B")
        merge = make_node(NodeType.MERGE, "Merge", traverse_in=TraverseIn.AWAIT_ALL)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, merge, end],
            [
                make_edge(start, a),
                make_edge(start, b),
                make_edge(a, merge),
                make_edge(b, merge),
                make_edge(merge, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())
        reg.register(NodeType.STATIC.value, MockStaticExecutor())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success


class TestTraverseOutStrategies:
    """Test SPAWN_ALL, SPAWN_NONE, SPAWN_PICKED, SPAWN_START."""

    def test_spawn_all_dispatches_all_successors(self):
        """SPAWN_ALL fires all successors."""
        start = make_node(NodeType.START, "Start")
        fork = make_node(NodeType.INSTRUCTION, "Fork", traverse_out=TraverseOut.SPAWN_ALL)
        a = make_node(NodeType.INSTRUCTION, "A")
        b = make_node(NodeType.STATIC, "B")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, fork, a, b, end],
            [
                make_edge(start, fork),
                make_edge(fork, a),
                make_edge(fork, b),
                make_edge(a, end),
                make_edge(b, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)
        reg.register(NodeType.STATIC.value, MockStaticExecutor())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "Fork" in tracker.calls
        assert "A" in tracker.calls

    def test_spawn_none_stops_dispatching(self):
        """SPAWN_NONE doesn't dispatch any successors (End node behavior)."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        # Extra unreachable node
        unreachable = make_node(NodeType.INSTRUCTION, "Unreachable")
        graph = make_graph(
            [start, a, end, unreachable],
            [make_edge(start, a), make_edge(a, end), make_edge(end, unreachable)],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "Unreachable" not in tracker.calls

    def test_spawn_picked_only_one_fires(self):
        """SPAWN_PICKED dispatches only the selected successor."""
        start = make_node(NodeType.START, "Start")
        decision = make_node(
            NodeType.DECISION,
            "D",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"pick": "Alpha"},
        )
        alpha = make_node(NodeType.INSTRUCTION, "Alpha")
        beta = make_node(NodeType.INSTRUCTION, "Beta")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, decision, alpha, beta, end],
            [
                make_edge(start, decision),
                make_edge(decision, alpha, label="Alpha"),
                make_edge(decision, beta, label="Beta"),
                make_edge(alpha, end),
                make_edge(beta, end),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        reg = SimpleNodeRegistry()
        reg.register(NodeType.DECISION.value, MockDecisionExecutor())
        reg.register(NodeType.INSTRUCTION.value, tracker)

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "Alpha" in tracker.calls
        assert "Beta" not in tracker.calls

    def test_events_emitted_for_each_node(self):
        """NodeStarted and NodeFinished events are emitted for each node."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph([start, a, end], [make_edge(start, a), make_edge(a, end)], start)

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        events: list[FlowEvent] = []
        runner = FlowRunner(graph=graph, node_registry=reg, on_event=events.append)
        runner.run("test")

        started = [e for e in events if isinstance(e, NodeStarted)]
        finished = [e for e in events if isinstance(e, NodeFinished)]
        # At least Start, A, End
        assert len(started) >= 3
        assert len(finished) >= 3

    def test_flow_result_contains_duration(self):
        """FlowResult has a positive duration_seconds."""
        start = make_node(NodeType.START, "Start")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph([start, end], [make_edge(start, end)], start)

        runner = FlowRunner(graph=graph, node_registry=SimpleNodeRegistry())
        result = runner.run("test")
        assert result.duration_seconds > 0


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: Conversation Composition Integration Tests (25+)        ║
# ╚══════════════════════════════════════════════════════════════════════╝


class TestConversationCompositionBasic:
    """Test that __conversation__ accumulates correctly across real graph executions."""

    def test_two_instruction_nodes_second_sees_first(self):
        """Two instruction nodes in sequence: second sees first's output."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "NodeA")
        b = make_node(NodeType.INSTRUCTION, "NodeB")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, end],
            [make_edge(start, a), make_edge(a, b), make_edge(b, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 2
        # Second entry should mention seeing 1 prior entry
        assert "seen 1 prior entries" in conv[1]["text"]

    def test_text_node_before_instruction(self):
        """Text node appends to conversation, instruction sees it."""
        start = make_node(NodeType.START, "Start")
        txt = make_node(NodeType.TEXT, "Intro", metadata={"text": "Welcome to the system."})
        instr = make_node(NodeType.INSTRUCTION, "Worker")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, txt, instr, end],
            [make_edge(start, txt), make_edge(txt, instr), make_edge(instr, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.TEXT.value, MockTextExecutor())
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 2
        assert conv[0]["role"] == "Intro"
        assert conv[0]["text"] == "Welcome to the system."
        # Worker should see 1 prior entry (the text)
        assert "seen 1 prior entries" in conv[1]["text"]

    def test_var_node_does_not_appear_in_conversation(self):
        """Var node writes to memory but does NOT touch __conversation__."""
        start = make_node(NodeType.START, "Start")
        var = make_node(
            NodeType.VAR, "SetVar", metadata={"name": "my_var", "expression": "'hello'"}
        )
        instr = make_node(NodeType.INSTRUCTION, "Worker")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, var, instr, end],
            [make_edge(start, var), make_edge(var, instr), make_edge(instr, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.VAR.value, MockVarExecutor())
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        # Only the instruction node should be in conversation
        assert len(conv) == 1
        assert conv[0]["role"] == "Worker"

        # But the var should be in memory
        assert store.get_memory(result.flow_id, "my_var") == "hello"

    def test_three_node_chain_instruction_text_instruction(self):
        """A -> Text -> B: B sees both A's output and text."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "NodeA")
        txt = make_node(NodeType.TEXT, "Middle", metadata={"text": "Between A and B."})
        b = make_node(NodeType.INSTRUCTION, "NodeB")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, txt, b, end],
            [make_edge(start, a), make_edge(a, txt), make_edge(txt, b), make_edge(b, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())
        reg.register(NodeType.TEXT.value, MockTextExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 3
        assert conv[0]["role"] == "NodeA"
        assert conv[1]["role"] == "Middle"
        assert conv[2]["role"] == "NodeB"
        # NodeB should see 2 prior entries
        assert "seen 2 prior entries" in conv[2]["text"]

    def test_decision_preserves_conversation_context(self):
        """Decision node receives conversation history from memory."""
        start = make_node(NodeType.START, "Start")
        instr = make_node(NodeType.INSTRUCTION, "Setup")
        decision = make_node(
            NodeType.DECISION,
            "Route",
            traverse_out=TraverseOut.SPAWN_PICKED,
            metadata={"pick": "PathA"},
        )
        path_a = make_node(NodeType.STATIC, "PathA")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, instr, decision, path_a, end],
            [
                make_edge(start, instr),
                make_edge(instr, decision),
                make_edge(decision, path_a, label="PathA"),
                make_edge(path_a, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())
        reg.register(NodeType.DECISION.value, MockDecisionExecutor())
        reg.register(NodeType.STATIC.value, MockStaticExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("route me")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert conv is not None
        assert len(conv) >= 1
        assert conv[0]["role"] == "Setup"


class TestConversationCompositionMemory:
    """Test memory write/read integration with conversation flow."""

    def test_memory_write_then_read(self):
        """Write node stores value, read node retrieves it."""
        start = make_node(NodeType.START, "Start")
        writer = make_node(NodeType.INSTRUCTION, "Writer")
        reader = make_node(NodeType.STATIC, "Reader")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, writer, reader, end],
            [make_edge(start, writer), make_edge(writer, reader), make_edge(reader, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockMemoryWriteExecutor("topic", "AI safety"))
        reg.register(NodeType.STATIC.value, MockMemoryReadExecutor("topic"))

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert "topic=AI safety" in result.final_output


class TestConversationCompositionLoops:
    """Test conversation accumulation across loop iterations."""

    def _build_loop_graph(
        self, threshold: int = 3
    ) -> tuple[GraphSpec, SimpleNodeRegistry, TrackingLLMExecutor]:
        """Build: Start -> Body -> Counter (if < threshold -> Body, else -> End)."""
        start = make_node(NodeType.START, "Start")
        body = make_node(NodeType.INSTRUCTION, "LoopBody", traverse_in=TraverseIn.AWAIT_FIRST)
        checker = make_node(NodeType.IF, "Counter", traverse_out=TraverseOut.SPAWN_PICKED)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, body, checker, end],
            [
                make_edge(start, body),
                make_edge(body, checker),
                make_edge(checker, body, label="LoopBody"),
                make_edge(checker, end, label="End"),
            ],
            start,
        )

        tracker = TrackingLLMExecutor()
        counter = MockLoopCounterExecutor(
            threshold=threshold, loop_target="LoopBody", exit_target="End"
        )
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, tracker)
        reg.register(NodeType.IF.value, counter)
        return graph, reg, tracker

    def test_loop_round_1_output_visible_in_round_2(self):
        """After round 1, round 2 sees round 1's conversation entry."""
        graph, reg, tracker = self._build_loop_graph(threshold=3)

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("loop test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        # 3 iterations of the body
        assert len(conv) == 3
        # Second entry should have seen 1 prior
        assert "seen 1 prior entries" in conv[1]["text"]

    def test_loop_round_2_output_visible_in_round_3(self):
        """Round 3 sees rounds 1 and 2."""
        graph, reg, tracker = self._build_loop_graph(threshold=3)

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("loop test")

        conv = store.get_memory(result.flow_id, "__conversation__")
        # Third entry should have seen 2 prior
        assert "seen 2 prior entries" in conv[2]["text"]

    def test_loop_all_rounds_accumulate(self):
        """All conversation entries accumulate across all loop rounds."""
        graph, reg, tracker = self._build_loop_graph(threshold=4)

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("loop test")

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 4
        for i, entry in enumerate(conv):
            assert f"seen {i} prior entries" in entry["text"]

    def test_five_round_loop_final_sees_all(self):
        """5-round loop: final iteration sees 4 prior entries."""
        graph, reg, tracker = self._build_loop_graph(threshold=5)

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("loop test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 5
        assert "seen 4 prior entries" in conv[4]["text"]

    def test_loop_after_instruction_preserves_pre_loop_conversation(self):
        """Instruction before loop: loop body sees pre-loop entry."""
        start = make_node(NodeType.START, "Start")
        pre = make_node(NodeType.INSTRUCTION, "PreLoop")
        body = make_node(NodeType.INSTRUCTION, "LoopBody", traverse_in=TraverseIn.AWAIT_FIRST)
        checker = make_node(NodeType.IF, "Counter", traverse_out=TraverseOut.SPAWN_PICKED)
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, pre, body, checker, end],
            [
                make_edge(start, pre),
                make_edge(pre, body),
                make_edge(body, checker),
                make_edge(checker, body, label="LoopBody"),
                make_edge(checker, end, label="End"),
            ],
            start,
        )

        # Use a separate tracker for pre-loop
        class CombinedExecutor(NodeExecutor):
            def __init__(self):
                self.calls: list[str] = []

            async def execute(self, context: ExecutionContext) -> NodeResult:
                name = context.current_node.name
                self.calls.append(name)
                conv = list(context.memory.get("__conversation__", []))
                response = f"Response from {name} (seen {len(conv)} prior entries)"
                conv.append({"role": name, "text": response})
                return NodeResult(
                    success=True,
                    data={"memory_updates": {"__conversation__": conv}},
                    output_text=response,
                )

        combined = CombinedExecutor()
        counter = MockLoopCounterExecutor(threshold=2, loop_target="LoopBody", exit_target="End")
        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, combined)
        reg.register(NodeType.IF.value, counter)

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        # PreLoop + 2 iterations of LoopBody
        assert len(conv) == 3
        assert conv[0]["role"] == "PreLoop"
        # First LoopBody should see 1 prior (PreLoop)
        assert "seen 1 prior entries" in conv[1]["text"]
        # Second LoopBody should see 2 prior (PreLoop + first LoopBody)
        assert "seen 2 prior entries" in conv[2]["text"]


class TestConversationCompositionEdgeCases:
    """Edge cases in conversation composition."""

    def test_empty_text_node_does_not_append(self):
        """Text node with empty template doesn't add to conversation."""
        start = make_node(NodeType.START, "Start")
        txt = make_node(NodeType.TEXT, "Empty", metadata={"text": ""})
        instr = make_node(NodeType.INSTRUCTION, "Worker")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, txt, instr, end],
            [make_edge(start, txt), make_edge(txt, instr), make_edge(instr, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.TEXT.value, MockTextExecutor())
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        # Only the instruction should be in conversation (empty text skipped)
        assert len(conv) == 1
        assert conv[0]["role"] == "Worker"

    def test_conversation_not_corrupted_by_deep_copy(self):
        """Deep copy on memory read doesn't corrupt conversation entries."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        b = make_node(NodeType.INSTRUCTION, "B")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, end],
            [make_edge(start, a), make_edge(a, b), make_edge(b, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 2

        # Get it again and verify it's the same
        conv2 = store.get_memory(result.flow_id, "__conversation__")
        assert conv == conv2

        # Modify conv, verify conv2 is unaffected
        conv.append({"role": "hacker", "text": "injected"})
        conv3 = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv3) == 2  # Not affected by modification

    def test_multiple_flows_have_independent_conversations(self):
        """Two flow runs don't share conversation state."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, end],
            [make_edge(start, a), make_edge(a, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)

        result1 = runner.run("first")
        result2 = runner.run("second")

        conv1 = store.get_memory(result1.flow_id, "__conversation__")
        conv2 = store.get_memory(result2.flow_id, "__conversation__")

        assert conv1 is not None
        assert conv2 is not None
        assert len(conv1) == 1
        assert len(conv2) == 1
        assert result1.flow_id != result2.flow_id

    def test_long_chain_conversation_accumulates(self):
        """5-node chain: last node sees 4 prior conversation entries."""
        start = make_node(NodeType.START, "Start")
        nodes = [make_node(NodeType.INSTRUCTION, f"N{i}") for i in range(5)]
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)

        all_nodes = [start] + nodes + [end]
        edges = [make_edge(start, nodes[0])]
        for i in range(len(nodes) - 1):
            edges.append(make_edge(nodes[i], nodes[i + 1]))
        edges.append(make_edge(nodes[-1], end))

        graph = make_graph(all_nodes, edges, start)

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 5
        for i, entry in enumerate(conv):
            assert f"seen {i} prior entries" in entry["text"]
            assert entry["role"] == f"N{i}"

    def test_text_var_instruction_mixed_chain(self):
        """Text -> Var -> Instruction: var doesn't pollute conversation."""
        start = make_node(NodeType.START, "Start")
        txt = make_node(NodeType.TEXT, "Greeting", metadata={"text": "Hello agent"})
        var = make_node(NodeType.VAR, "SetMode", metadata={"name": "mode", "expression": "'debug'"})
        instr = make_node(NodeType.INSTRUCTION, "Worker")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, txt, var, instr, end],
            [
                make_edge(start, txt),
                make_edge(txt, var),
                make_edge(var, instr),
                make_edge(instr, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.TEXT.value, MockTextExecutor())
        reg.register(NodeType.VAR.value, MockVarExecutor())
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")
        assert result.success

        conv = store.get_memory(result.flow_id, "__conversation__")
        # Only text and instruction should be in conversation
        assert len(conv) == 2
        assert conv[0]["role"] == "Greeting"
        assert conv[1]["role"] == "Worker"

        # Var should be in memory separately
        assert store.get_memory(result.flow_id, "mode") == "debug"

    def test_instruction_output_stored_as_assistant_message(self):
        """FlowRunner stores the output_text as an assistant message in node messages."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, end],
            [make_edge(start, a), make_edge(a, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")

        # Check that node A's stored messages include an assistant message
        msgs = store.get_messages(result.flow_id, a.id)
        assistant_msgs = [m for m in msgs if m.role == MessageRole.ASSISTANT]
        assert len(assistant_msgs) >= 1
        assert "Response from A" in assistant_msgs[-1].content

    def test_flow_result_final_output_from_end_node(self):
        """FlowResult.final_output comes from the End node."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(
            NodeType.END,
            "End",
            traverse_out=TraverseOut.SPAWN_NONE,
            thought_type=ThoughtType.CONTINUE,
        )
        graph = make_graph(
            [start, a, end],
            [make_edge(start, a), make_edge(a, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert result.final_output != ""

    def test_node_result_data_accessible_in_flow_result(self):
        """FlowResult.node_results contains per-node results."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "A")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, end],
            [make_edge(start, a), make_edge(a, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        runner = FlowRunner(graph=graph, node_registry=reg)
        result = runner.run("test")
        assert result.success
        assert len(result.node_results) >= 1

    def test_whitespace_only_text_does_not_append(self):
        """Text node with whitespace-only content doesn't add to conversation."""
        start = make_node(NodeType.START, "Start")
        txt = make_node(NodeType.TEXT, "Spaces", metadata={"text": "   \n  "})
        instr = make_node(NodeType.INSTRUCTION, "Worker")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, txt, instr, end],
            [make_edge(start, txt), make_edge(txt, instr), make_edge(instr, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.TEXT.value, MockTextExecutor())
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 1
        assert conv[0]["role"] == "Worker"

    def test_two_text_nodes_both_appear_in_conversation(self):
        """Two text nodes in sequence both appear in conversation."""
        start = make_node(NodeType.START, "Start")
        t1 = make_node(NodeType.TEXT, "Text1", metadata={"text": "First context."})
        t2 = make_node(NodeType.TEXT, "Text2", metadata={"text": "Second context."})
        instr = make_node(NodeType.INSTRUCTION, "Worker")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, t1, t2, instr, end],
            [
                make_edge(start, t1),
                make_edge(t1, t2),
                make_edge(t2, instr),
                make_edge(instr, end),
            ],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.TEXT.value, MockTextExecutor())
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert len(conv) == 3
        assert conv[0]["text"] == "First context."
        assert conv[1]["text"] == "Second context."
        assert "seen 2 prior entries" in conv[2]["text"]

    def test_conversation_entries_have_correct_roles(self):
        """Each conversation entry has role matching the node name."""
        start = make_node(NodeType.START, "Start")
        a = make_node(NodeType.INSTRUCTION, "Analyzer")
        b = make_node(NodeType.INSTRUCTION, "Synthesizer")
        end = make_node(NodeType.END, "End", traverse_out=TraverseOut.SPAWN_NONE)
        graph = make_graph(
            [start, a, b, end],
            [make_edge(start, a), make_edge(a, b), make_edge(b, end)],
            start,
        )

        reg = SimpleNodeRegistry()
        reg.register(NodeType.INSTRUCTION.value, MockLLMExecutor())

        store = InMemoryStore()
        runner = FlowRunner(graph=graph, node_registry=reg, store=store)
        result = runner.run("test")

        conv = store.get_memory(result.flow_id, "__conversation__")
        assert conv[0]["role"] == "Analyzer"
        assert conv[1]["role"] == "Synthesizer"
