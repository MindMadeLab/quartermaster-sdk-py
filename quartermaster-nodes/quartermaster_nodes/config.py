"""Configuration models for nodes and flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set
from uuid import UUID

from quartermaster_nodes.enums import (
    AvailableErrorHandlingStrategies,
    AvailableMessageTypes,
    AvailableThoughtTypes,
    AvailableTraversingIn,
    AvailableTraversingOut,
)


class AssistantInfo:
    """Metadata about a node type."""

    version: str = ""
    description: str = ""
    instructions: str = ""
    metadata: dict = {}


class FlowNodeConf:
    """Defines a node's flow behavior constraints."""

    def __init__(
        self,
        traverse_in: AvailableTraversingIn,
        traverse_out: AvailableTraversingOut,
        thought_type: AvailableThoughtTypes,
        message_type: AvailableMessageTypes,
        error_handling_strategy: AvailableErrorHandlingStrategies = AvailableErrorHandlingStrategies.Stop,
        available_thought_types: Set[AvailableThoughtTypes] | None = None,
        available_traversing_in: Set[AvailableTraversingIn] | None = None,
        available_traversing_out: Set[AvailableTraversingOut] | None = None,
        available_message_types: Set[AvailableMessageTypes] | None = None,
        available_error_handling_strategies: Set[AvailableErrorHandlingStrategies] | None = None,
        accepts_incoming_edges: bool = True,
        accepts_outgoing_edges: bool = True,
    ) -> None:
        self.traverse_in = traverse_in
        self.traverse_out = traverse_out
        self.thought_type = thought_type
        self.message_type = message_type
        self.error_handling_strategy = error_handling_strategy

        self.available_thought_types = (available_thought_types or set()).copy()
        self.available_traversing_in = (available_traversing_in or set()).copy()
        self.available_traversing_out = (available_traversing_out or set()).copy()
        self.available_message_types = (available_message_types or set()).copy()

        if available_error_handling_strategies is None:
            self.available_error_handling_strategies = {
                AvailableErrorHandlingStrategies.Stop,
                AvailableErrorHandlingStrategies.Continue,
            }
        else:
            self.available_error_handling_strategies = available_error_handling_strategies.copy()

        self.accepts_incoming_edges = accepts_incoming_edges
        self.accepts_outgoing_edges = accepts_outgoing_edges

        # Auto-add defaults to available sets
        self.available_thought_types.add(thought_type)
        self.available_traversing_in.add(traverse_in)
        self.available_traversing_out.add(traverse_out)
        self.available_message_types.add(message_type)
        self.available_error_handling_strategies.add(error_handling_strategy)

    def asdict(self) -> dict:
        """Serialize to dict with enum names."""
        return {
            "traverse_in": self.traverse_in.name,
            "traverse_out": self.traverse_out.name,
            "thought_type": self.thought_type.name,
            "message_type": self.message_type.name,
            "error_handling_strategy": self.error_handling_strategy.name,
            "accepts_incoming_edges": self.accepts_incoming_edges,
            "accepts_outgoing_edges": self.accepts_outgoing_edges,
            "available_thought_types": sorted(t.name for t in self.available_thought_types),
            "available_traversing_in": sorted(t.name for t in self.available_traversing_in),
            "available_traversing_out": sorted(t.name for t in self.available_traversing_out),
            "available_message_types": sorted(t.name for t in self.available_message_types),
            "available_error_handling_strategies": sorted(
                s.name for s in self.available_error_handling_strategies
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> FlowNodeConf:
        """Deserialize from dict with enum names."""
        return FlowNodeConf(
            traverse_in=AvailableTraversingIn[config_dict["traverse_in"]],
            traverse_out=AvailableTraversingOut[config_dict["traverse_out"]],
            thought_type=AvailableThoughtTypes[config_dict["thought_type"]],
            message_type=AvailableMessageTypes[config_dict["message_type"]],
            error_handling_strategy=AvailableErrorHandlingStrategies[
                config_dict.get("error_handling_strategy", "Stop")
            ],
        )

    def is_valid_thought_type(self, thought_type: str) -> bool:
        return AvailableThoughtTypes[thought_type] in self.available_thought_types

    def is_valid_traversing_in(self, traversing_in: str) -> bool:
        return AvailableTraversingIn[traversing_in] in self.available_traversing_in

    def is_valid_traversing_out(self, traversing_out: str) -> bool:
        return AvailableTraversingOut[traversing_out] in self.available_traversing_out

    def is_valid_message_type(self, message_type: str) -> bool:
        return AvailableMessageTypes[message_type] in self.available_message_types

    def is_valid_error_handling_strategy(self, error_handling_strategy: Optional[str]) -> bool:
        if error_handling_strategy is None:
            return True
        return (
            AvailableErrorHandlingStrategies[error_handling_strategy]
            in self.available_error_handling_strategies
        )

    def validate(self) -> None:
        """Validate that current config values are within allowed sets."""
        if not self.is_valid_thought_type(self.thought_type.name):
            raise ValueError(f"Invalid thought type: {self.thought_type.name}")
        if not self.is_valid_traversing_in(self.traverse_in.name):
            raise ValueError(f"Invalid traversing in: {self.traverse_in.name}")
        if not self.is_valid_traversing_out(self.traverse_out.name):
            raise ValueError(f"Invalid traversing out: {self.traverse_out.name}")
        if not self.is_valid_message_type(self.message_type.name):
            raise ValueError(f"Invalid message type: {self.message_type.name}")
        if not self.is_valid_error_handling_strategy(self.error_handling_strategy.name):
            raise ValueError(
                f"Invalid error handling strategy: {self.error_handling_strategy.name}"
            )


@dataclass
class FlowRunConfig:
    """Runtime context for a flow execution."""

    assistant_node_id: UUID
    user_id: UUID
    chat_id: UUID
    environment_id: Optional[UUID] = None
    previous_thought_id: Optional[UUID] = None
    parent_flow_node_id: Optional[UUID] = None
    scheduled: bool = False

    def copy(self) -> FlowRunConfig:
        return FlowRunConfig(
            assistant_node_id=self.assistant_node_id,
            user_id=self.user_id,
            chat_id=self.chat_id,
            environment_id=self.environment_id,
            previous_thought_id=self.previous_thought_id,
            parent_flow_node_id=self.parent_flow_node_id,
            scheduled=self.scheduled,
        )

    def asdict(self) -> dict:
        """Serialize UUIDs to strings for JSON serialization."""
        return {
            "assistant_node_id": str(self.assistant_node_id)
            if self.assistant_node_id is not None
            else None,
            "user_id": str(self.user_id) if self.user_id is not None else None,
            "chat_id": str(self.chat_id) if self.chat_id is not None else None,
            "environment_id": str(self.environment_id)
            if self.environment_id is not None
            else None,
            "previous_thought_id": str(self.previous_thought_id)
            if self.previous_thought_id is not None
            else None,
            "parent_flow_node_id": str(self.parent_flow_node_id)
            if self.parent_flow_node_id is not None
            else None,
            "scheduled": self.scheduled,
        }

    @staticmethod
    def from_dict(config_dict: dict) -> FlowRunConfig:
        """Parse strings back into UUIDs."""
        return FlowRunConfig(
            assistant_node_id=UUID(config_dict["assistant_node_id"])
            if config_dict.get("assistant_node_id") is not None
            else None,  # type: ignore[arg-type]
            user_id=UUID(config_dict["user_id"])
            if config_dict.get("user_id") is not None
            else None,  # type: ignore[arg-type]
            chat_id=UUID(config_dict["chat_id"])
            if config_dict.get("chat_id") is not None
            else None,  # type: ignore[arg-type]
            environment_id=UUID(config_dict["environment_id"])
            if config_dict.get("environment_id") is not None
            else None,
            previous_thought_id=UUID(config_dict["previous_thought_id"])
            if config_dict.get("previous_thought_id") is not None
            else None,
            parent_flow_node_id=UUID(config_dict["parent_flow_node_id"])
            if config_dict.get("parent_flow_node_id") is not None
            else None,
            scheduled=config_dict.get("scheduled", False),
        )

    def __str__(self) -> str:
        return (
            f"assistant node id: {self.assistant_node_id}\n"
            f"user id: {self.user_id}\n"
            f"chat id: {self.chat_id}\n"
            f"previous thought id: {self.previous_thought_id}\n"
            f"parent flow node id: {self.parent_flow_node_id}\n"
            f"scheduled: {self.scheduled}"
        )
