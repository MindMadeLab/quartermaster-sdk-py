"""Message routing — manages conversation history flow between nodes.

When a flow traverses from node A to node B, the message router determines
what conversation history node B should receive based on the ThoughtType
and MessageType configurations.
"""

from __future__ import annotations

from uuid import UUID

from qm_engine.stores.base import ExecutionStore
from qm_engine.types import (
    AgentVersion,
    GraphNode,
    Message,
    MessageRole,
    MessageType,
    ThoughtType,
)


class MessageRouter:
    """Routes messages between nodes during flow execution.

    Handles the thought system: whether a node gets a fresh conversation,
    inherits from its predecessor, or continues with accumulated messages.
    """

    def __init__(self, store: ExecutionStore) -> None:
        self._store = store

    def get_messages_for_node(
        self,
        flow_id: UUID,
        node: GraphNode,
        graph: AgentVersion,
    ) -> list[Message]:
        """Build the message list a node should receive.

        Args:
            flow_id: The current flow execution ID.
            node: The node about to execute.
            graph: The agent graph.

        Returns:
            The conversation history this node should process.
        """
        thought_type = node.thought_type

        if thought_type == ThoughtType.SKIP:
            return []

        if thought_type == ThoughtType.NEW:
            return self._build_new_thought(flow_id, node)

        if thought_type == ThoughtType.NEW_HIDDEN:
            return self._build_new_thought(flow_id, node)

        if thought_type == ThoughtType.INHERIT:
            return self._inherit_from_predecessors(flow_id, node, graph)

        if thought_type == ThoughtType.CONTINUE:
            return self._continue_conversation(flow_id, node, graph)

        return []

    def save_node_output(
        self,
        flow_id: UUID,
        node_id: UUID,
        messages: list[Message],
    ) -> None:
        """Save the messages produced by a node after execution."""
        self._store.save_messages(flow_id, node_id, messages)

    def append_to_node(
        self,
        flow_id: UUID,
        node_id: UUID,
        message: Message,
    ) -> None:
        """Append a single message to a node's history."""
        self._store.append_message(flow_id, node_id, message)

    def _build_new_thought(self, flow_id: UUID, node: GraphNode) -> list[Message]:
        """Start a fresh conversation, optionally with a system instruction."""
        messages: list[Message] = []
        system_instruction = node.metadata.get("system_instruction", "")
        if system_instruction:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_instruction))
        return messages

    def _inherit_from_predecessors(
        self,
        flow_id: UUID,
        node: GraphNode,
        graph: AgentVersion,
    ) -> list[Message]:
        """Inherit the last message from each predecessor."""
        predecessors = graph.get_predecessors(node.id)
        messages: list[Message] = []

        # Add system instruction if present
        system_instruction = node.metadata.get("system_instruction", "")
        if system_instruction:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_instruction))

        # Collect last message from each predecessor
        for pred in predecessors:
            pred_messages = self._store.get_messages(flow_id, pred.id)
            if pred_messages:
                last = pred_messages[-1]
                messages.append(last)

        return messages

    def _continue_conversation(
        self,
        flow_id: UUID,
        node: GraphNode,
        graph: AgentVersion,
    ) -> list[Message]:
        """Continue with the full accumulated conversation history."""
        predecessors = graph.get_predecessors(node.id)
        messages: list[Message] = []

        # Add system instruction if present
        system_instruction = node.metadata.get("system_instruction", "")
        if system_instruction:
            messages.append(Message(role=MessageRole.SYSTEM, content=system_instruction))

        # Gather all messages from predecessors (merge histories)
        for pred in predecessors:
            pred_messages = self._store.get_messages(flow_id, pred.id)
            messages.extend(pred_messages)

        return messages

    def build_input_message(
        self,
        node: GraphNode,
        user_input: str,
        memory: dict[str, object],
    ) -> Message | None:
        """Build the input message for a node based on its MessageType.

        Args:
            node: The node about to execute.
            user_input: The original user input to the flow.
            memory: Flow-scoped memory variables.

        Returns:
            A Message to prepend, or None if no input message is needed.
        """
        msg_type = node.message_type

        if msg_type == MessageType.USER:
            return Message(role=MessageRole.USER, content=user_input)

        if msg_type == MessageType.VARIABLE:
            var_name = node.metadata.get("variable_name", "")
            var_value = memory.get(var_name, "")
            return Message(role=MessageRole.USER, content=str(var_value))

        if msg_type == MessageType.ASSISTANT:
            content = node.metadata.get("assistant_message", "")
            return Message(role=MessageRole.ASSISTANT, content=content)

        # AUTOMATIC — engine decides (usually user input for first node, inherit for rest)
        return None
