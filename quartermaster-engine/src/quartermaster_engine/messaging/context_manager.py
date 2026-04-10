"""Context window management — truncation strategies for LLM context limits.

When conversation history grows too large for an LLM's context window, the
ContextManager truncates messages using configurable strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from quartermaster_engine.types import Message, MessageRole


class TruncationStrategy(str, Enum):
    """Strategy for truncating messages when they exceed the context window."""

    DROP_OLDEST = "drop_oldest"
    DROP_MIDDLE = "drop_middle"
    SUMMARIZE = "summarize"


class ContextManager:
    """Manages message truncation to fit within LLM context windows.

    Ensures that the system message is always preserved and that the most
    recent messages are kept (since they have the most context).
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        max_messages: int | None = None,
        strategy: TruncationStrategy = TruncationStrategy.DROP_OLDEST,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize the context manager.

        Args:
            max_tokens: Maximum total tokens allowed.
            max_messages: Maximum number of messages allowed (None = no limit).
            strategy: How to truncate when limits are exceeded.
            token_counter: Function that estimates token count for a string.
                           Defaults to a simple word-based approximation.
        """
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.strategy = strategy
        self._count_tokens = token_counter or self._default_token_counter

    def truncate(self, messages: list[Message]) -> list[Message]:
        """Truncate messages to fit within configured limits.

        Preserves:
        - The system message (always first)
        - The most recent messages (most contextually relevant)

        Drops messages from the middle/oldest first.
        """
        if not messages:
            return messages

        # Apply message count limit first
        if self.max_messages and len(messages) > self.max_messages:
            messages = self._truncate_by_count(messages)

        # Apply token limit
        total = self._total_tokens(messages)
        if total > self.max_tokens:
            messages = self._truncate_by_tokens(messages)

        return messages

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate the total token count for a list of messages."""
        return self._total_tokens(messages)

    def _truncate_by_count(self, messages: list[Message]) -> list[Message]:
        """Truncate to max_messages, preserving system message and recent messages."""
        assert self.max_messages is not None
        if len(messages) <= self.max_messages:
            return messages

        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        keep_count = self.max_messages - len(system_msgs)
        if keep_count <= 0:
            return system_msgs[: self.max_messages]

        return system_msgs + non_system[-keep_count:]

    def _truncate_by_tokens(self, messages: list[Message]) -> list[Message]:
        """Truncate to fit within max_tokens, preserving system and recent messages."""
        # Separate system messages
        system_msgs = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        system_tokens = self._total_tokens(system_msgs)
        remaining_budget = self.max_tokens - system_tokens

        if remaining_budget <= 0:
            # Even system messages exceed the budget — truncate system messages
            return self._fit_to_budget(system_msgs, self.max_tokens)

        # Keep as many recent messages as fit in the remaining budget
        kept: list[Message] = []
        used = 0
        for msg in reversed(non_system):
            msg_tokens = self._count_tokens(msg.content)
            if used + msg_tokens > remaining_budget:
                break
            kept.insert(0, msg)
            used += msg_tokens

        return system_msgs + kept

    def _fit_to_budget(self, messages: list[Message], budget: int) -> list[Message]:
        """Keep messages from the end that fit within the token budget."""
        kept: list[Message] = []
        used = 0
        for msg in reversed(messages):
            msg_tokens = self._count_tokens(msg.content)
            if used + msg_tokens > budget:
                break
            kept.insert(0, msg)
            used += msg_tokens
        return kept

    def _total_tokens(self, messages: list[Message]) -> int:
        """Sum up estimated tokens across all messages."""
        return sum(self._count_tokens(m.content) for m in messages)

    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Rough token estimate: ~4 characters per token (conservative)."""
        return max(1, len(text) // 4)
