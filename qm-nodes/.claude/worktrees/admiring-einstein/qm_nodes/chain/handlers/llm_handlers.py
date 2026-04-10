"""LLM chain handlers for processing LLM requests.

These handlers form the building blocks of LLM processing pipelines.
Each handler handles one step of the LLM request lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from qm_nodes.chain.handler import Handler
from qm_nodes.protocols import (
    ContextManagerConfig,
    LLMConfig,
    LLMProvider,
    ProgramContainer,
)

logger = logging.getLogger(__name__)


class ValidateMemoryID(Handler):
    """Validates that a memory_id exists in the data."""

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        memory_id = data.get("memory_id")
        if memory_id is None:
            raise ValueError("memory_id is required but was None")
        return data


class PrepareMessages(Handler):
    """Loads and prepares messages from the conversation history.

    This handler loads messages from the thought/memory system and
    prepares them for LLM consumption. It supports adding additional
    messages (e.g., for decision nodes).
    """

    def __init__(
        self,
        client: Any,
        llm_config: LLMConfig,
        additional_message: Optional[str] = None,
        additional_message_role: Optional[str] = None,
    ) -> None:
        self.client = client
        self.llm_config = llm_config
        self.additional_message = additional_message
        self.additional_message_role = additional_message_role

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ctx = data.get("ctx")
        messages: list = data.get("messages", [])

        # If a message loader is provided in the context, use it
        message_loader = data.get("message_loader")
        if message_loader is not None:
            messages = message_loader(data["memory_id"], self.llm_config)

        # Add system message if configured
        if self.llm_config.system_message:
            messages.insert(0, {
                "role": "system",
                "content": self.llm_config.system_message,
            })

        # Add additional message if provided
        if self.additional_message:
            role = self.additional_message_role or "user"
            messages.append({
                "role": role,
                "content": self.additional_message,
            })

        data["messages"] = messages
        return data


class ContextManager(Handler):
    """Manages context window by truncating messages to fit token limits.

    Handles message truncation, tool result clearing, and image clearing
    based on the configured thresholds.
    """

    def __init__(
        self,
        client: Any,
        llm_config: LLMConfig,
        context_config: Optional[ContextManagerConfig] = None,
    ) -> None:
        self.client = client
        self.llm_config = llm_config
        self.context_config = context_config or ContextManagerConfig()

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        messages = data.get("messages", [])

        # Apply max messages limit
        if self.context_config.max_messages > 0 and len(messages) > self.context_config.max_messages:
            # Keep system message + last N messages
            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]
            messages = system_msgs + non_system[-self.context_config.max_messages :]

        data["messages"] = messages
        return data


class TransformToProvider(Handler):
    """Transforms internal messages to provider-specific format.

    Different LLM providers (OpenAI, Anthropic, etc.) expect different
    message formats. This handler applies the appropriate transformation.
    """

    def __init__(self, transformer: Optional[Callable] = None) -> None:
        self.transformer = transformer

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.transformer is not None:
            messages = data.get("messages", [])
            data["messages"] = self.transformer(messages)
        return data


class GenerateStreamResponse(Handler):
    """Calls the LLM provider to generate a streaming text response."""

    def __init__(
        self,
        client: Any,
        llm_config: LLMConfig,
        username: str = "",
    ) -> None:
        self.client = client
        self.llm_config = llm_config
        self.username = username

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        messages = data.get("messages", [])

        if hasattr(self.client, "generate_stream"):
            response = self.client.generate_stream(
                messages=messages,
                config=self.llm_config,
                username=self.username,
            )
            data["response"] = response
        else:
            logger.warning("Client does not support generate_stream")

        return data


class GenerateToolCall(Handler):
    """Generates a structured tool call response from the LLM.

    Used by decision nodes and other nodes that need the LLM to
    choose from a set of tools/functions.
    """

    def __init__(
        self,
        client: Any,
        tools: List[ProgramContainer],
        llm_config: LLMConfig,
        username: str = "",
        only_first_tool: bool = False,
    ) -> None:
        self.client = client
        self.tools = tools
        self.llm_config = llm_config
        self.username = username
        self.only_first_tool = only_first_tool

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        messages = data.get("messages", [])

        tool_dicts = [t.to_dict() if hasattr(t, "to_dict") else t for t in self.tools]

        if hasattr(self.client, "generate_structured"):
            response = self.client.generate_structured(
                messages=messages,
                tools=tool_dicts,
                config=self.llm_config,
                username=self.username,
            )
            data["response"] = response
            data["only_first_tool"] = self.only_first_tool

        return data


class GenerateNativeResponse(Handler):
    """Generates a response with automatic tool choice.

    The LLM decides whether to generate text or call tools based
    on the conversation context. Used by agent nodes.
    """

    def __init__(
        self,
        client: Any,
        tools: List[ProgramContainer],
        llm_config: LLMConfig,
        username: str = "",
    ) -> None:
        self.client = client
        self.tools = tools
        self.llm_config = llm_config
        self.username = username

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        messages = data.get("messages", [])

        tool_dicts = [t.to_dict() if hasattr(t, "to_dict") else t for t in self.tools]

        if hasattr(self.client, "generate_native"):
            response = self.client.generate_native(
                messages=messages,
                tools=tool_dicts,
                config=self.llm_config,
                username=self.username,
            )
            data["response"] = response

        return data


class ProcessStreamResponse(Handler):
    """Processes the streaming response from the LLM.

    Handles token accumulation, tool call parsing, and broadcasting
    updates to the client.
    """

    def __init__(
        self,
        memory_id_key: str = "memory_id",
        only_first_tool: bool = False,
    ) -> None:
        self.memory_id_key = memory_id_key
        self.only_first_tool = only_first_tool

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        response = data.get("response")
        ctx = data.get("ctx")

        if response is None:
            return data

        # Process the response - the actual implementation depends on the runtime
        response_processor = data.get("response_processor")
        if response_processor is not None:
            result = response_processor(
                response=response,
                memory_id=data.get(self.memory_id_key),
                ctx=ctx,
                only_first_tool=data.get("only_first_tool", self.only_first_tool),
            )
            data["processed_response"] = result
        elif hasattr(response, "text"):
            # Simple response object with text attribute
            data["processed_response"] = {
                "text": response.text,
                "tool_calls": getattr(response, "tool_calls", []),
                "usage": getattr(response, "usage", {}),
            }

        return data


class CaptureResponse(Handler):
    """Captures response tokens and tool calls without streaming.

    Used for non-streaming LLM calls where the full response is
    available at once.
    """

    def handle(self, data: Dict[str, Any]) -> Dict[str, Any]:
        response = data.get("response")
        if response is None:
            return data

        if hasattr(response, "text"):
            data["captured_text"] = response.text
        if hasattr(response, "tool_calls"):
            data["captured_tool_calls"] = response.tool_calls
        if hasattr(response, "usage"):
            data["captured_usage"] = response.usage

        return data
