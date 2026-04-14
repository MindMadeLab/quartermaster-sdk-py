"""Example runner — bridges Graph -> FlowRunner -> LLM providers.

Usage::

    from quartermaster_engine import run_graph

    agent = Graph("My Agent").start().user("Hi").instruction("Reply", ...).end()
    result = run_graph(agent, user_input="Tell me about AI")
    print(result.final_output)

Auto-detects API keys from environment (or .env file at CWD):
    ANTHROPIC_API_KEY  ->  claude-haiku-4-5-20251001
    OPENAI_API_KEY     ->  gpt-4o

Or force a provider:
    result = run_graph(agent, user_input="...", provider="anthropic")
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from typing import Any

# Load .env from current working directory if it exists
_env_file = pathlib.Path.cwd() / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip("'\"")
            if _key:
                os.environ[_key] = _val

from quartermaster_engine.runner.flow_runner import FlowRunner, FlowResult
from quartermaster_engine.stores.memory_store import InMemoryStore
from quartermaster_engine.events import NodeStarted, NodeFinished, FlowEvent, FlowError, TokenGenerated
from quartermaster_engine.nodes import SimpleNodeRegistry, NodeExecutor, NodeResult
from quartermaster_engine.context.execution_context import ExecutionContext
from quartermaster_graph import AgentGraph
from quartermaster_graph.enums import NodeType
from quartermaster_providers import ProviderRegistry, LLMConfig


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

_MODEL_MAP = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "xai": "grok-3-mini-fast",
    "google": "gemini-2.0-flash",
}


def _detect_provider() -> tuple[str, str]:
    """Auto-detect provider from environment variables."""
    for provider, model in _MODEL_MAP.items():
        env_key = f"{provider.upper()}_API_KEY"
        if os.environ.get(env_key):
            return provider, model
    return "", ""


_PROVIDER_CLASSES = {
    "anthropic": "quartermaster_providers.providers.anthropic:AnthropicProvider",
    "openai": "quartermaster_providers.providers.openai:OpenAIProvider",
    "google": "quartermaster_providers.providers.google:GoogleProvider",
    "groq": "quartermaster_providers.providers.groq:GroqProvider",
    "xai": "quartermaster_providers.providers.xai:XAIProvider",
}


def _get_provider_registry(provider_name: str) -> ProviderRegistry:
    """Create a ProviderRegistry with ALL available providers (based on env keys)."""
    import importlib
    registry = ProviderRegistry()
    registered = []
    for name, cls_path in _PROVIDER_CLASSES.items():
        api_key = os.environ.get(f"{name.upper()}_API_KEY", "")
        if not api_key:
            continue
        module_path, cls_name = cls_path.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            provider_cls = getattr(module, cls_name)
            registry.register(name, provider_cls, api_key=api_key)
            registered.append(name)
        except ImportError:
            pass  # SDK not installed for this provider
    if not registered:
        raise ValueError("No provider SDKs installed. pip install anthropic / openai / etc.")
    return registry


# ---------------------------------------------------------------------------
# Node executors
# ---------------------------------------------------------------------------

class LLMExecutor(NodeExecutor):
    """Calls a real LLM for Instruction and Summarize nodes."""

    def __init__(self, provider_registry: ProviderRegistry, default_model: str, default_provider: str):
        self._registry = provider_registry
        self._default_model = default_model
        self._default_provider = default_provider

    async def execute(self, context: ExecutionContext) -> NodeResult:
        system_instruction = context.get_meta("llm_system_instruction", "")
        model = context.get_meta("llm_model", "") or self._default_model
        provider_name = context.get_meta("llm_provider", "") or self._default_provider

        # Fall back to default if requested provider isn't registered
        try:
            provider = self._registry.get(provider_name)
        except Exception:
            provider_name = self._default_provider
            model = self._default_model
            try:
                provider = self._registry.get(provider_name)
            except Exception:
                return NodeResult(success=False, data={}, error=f"Provider '{provider_name}' not available")

        # Build prompt from conversation history stored in memory
        conversation = context.memory.get("__conversation__", [])
        user_input = str(context.memory.get("__user_input__", "Hello"))

        # Also check engine messages
        for msg in context.messages:
            if msg.content:
                user_input = msg.content

        # Build the prompt: include recent conversation for context (truncated)
        if conversation:
            parts = []
            for entry in conversation[-6:]:
                text = entry["text"]
                # Truncate each entry to ~500 chars to avoid context overflow
                if len(text) > 500:
                    text = text[:500] + "..."
                parts.append(f"[{entry['role']}]: {text}")
            history = "\n\n".join(parts)
            prompt = f"Previous conversation:\n{history}\n\nUser's original request: {user_input}"
        else:
            prompt = user_input

        config = LLMConfig(
            model=model,
            provider=provider_name,
            system_message=system_instruction,
            temperature=context.get_meta("llm_temperature", 0.7),
            stream=False,
        )

        try:
            response = await provider.generate_text_response(prompt, config)
            text = response.content
            context.emit_token(text)

            # Append to conversation history in memory
            node_name = context.current_node.name if context.current_node else "Assistant"
            conversation.append({"role": node_name, "text": text})
            return NodeResult(
                success=True, data={"memory_updates": {"__conversation__": conversation}},
                output_text=text,
            )
        except Exception as e:
            print(f"  [LLM ERROR] {provider_name}/{model}: {e}", flush=True)
            return NodeResult(success=False, data={}, error=str(e))


class DecisionExecutor(NodeExecutor):
    """Calls LLM to pick one branch for Decision nodes."""

    def __init__(self, provider_registry: ProviderRegistry, default_model: str, default_provider: str):
        self._registry = provider_registry
        self._default_model = default_model
        self._default_provider = default_provider

    async def execute(self, context: ExecutionContext) -> NodeResult:
        system_instruction = context.get_meta("llm_system_instruction", "")
        model = context.get_meta("llm_model", "") or self._default_model
        provider_name = context.get_meta("llm_provider", "") or self._default_provider

        # Get available options from outgoing edges
        edges = context.graph.get_edges_from(context.node_id)
        options = [e.label for e in edges if e.label]

        decision_prompt = system_instruction or "Choose the most appropriate option."
        if options:
            decision_prompt += f"\n\nOptions: {', '.join(options)}\nRespond with EXACTLY one of the options above, nothing else."

        # Fall back to default if requested provider isn't registered
        try:
            provider = self._registry.get(provider_name)
        except Exception:
            provider_name = self._default_provider
            model = self._default_model
            try:
                provider = self._registry.get(provider_name)
            except Exception:
                # Fallback: pick first option
                picked = options[0] if options else ""
                return NodeResult(success=True, data={}, output_text=picked, picked_node=picked)

        # Build prompt with conversation history
        conversation = context.memory.get("__conversation__", [])
        user_input = str(context.memory.get("__user_input__", "Choose"))
        for msg in context.messages:
            if msg.content:
                user_input = msg.content

        if conversation:
            history = "\n\n".join(
                f"[{entry['role']}] {entry['text']}"
                for entry in conversation[-4:]
            )
            prompt = f"Based on the conversation:\n{history}\n\nOriginal request: {user_input}"
        else:
            prompt = user_input

        config = LLMConfig(
            model=model,
            provider=provider_name,
            system_message=decision_prompt,
            temperature=0.0,
            stream=False,
        )

        try:
            response = await provider.generate_text_response(prompt, config)
            picked = response.content.strip()
            # Fuzzy match against options
            for opt in options:
                if opt.lower() in picked.lower():
                    picked = opt
                    break
            return NodeResult(success=True, data={}, output_text=picked, picked_node=picked)
        except Exception as e:
            picked = options[0] if options else ""
            return NodeResult(success=True, data={}, output_text=picked, picked_node=picked)


class UserExecutor(NodeExecutor):
    """Auto-provides user input (from the flow's initial input)."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        # In automated mode, pass through the user's input
        user_text = context.memory.get("__user_input__", "")
        return NodeResult(success=True, data={}, output_text=str(user_text))


class StaticExecutor(NodeExecutor):
    """Returns static text from node metadata."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        text = context.get_meta("static_text", "")
        return NodeResult(success=True, data={}, output_text=text)


class VarExecutor(NodeExecutor):
    """Sets a variable in flow memory, with expression evaluation."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        # Builder stores var name as "name" in metadata
        variable = context.get_meta("name", "") or context.get_meta("variable", "")
        expression = context.get_meta("expression", "")
        if variable:
            if expression:
                # Try to evaluate expression against flow memory
                try:
                    value = eval(expression, {"__builtins__": {}}, dict(context.memory))
                except Exception:
                    value = expression
            else:
                # No expression — capture last message content
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


class IfExecutor(NodeExecutor):
    """Evaluates a boolean expression and picks the true/false branch."""

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


class TextExecutor(NodeExecutor):
    """Renders a Jinja2 template using flow memory."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        template_str = context.get_meta("text", "")
        try:
            from jinja2 import Template
            template = Template(template_str)
            result = template.render(**context.memory)
        except Exception:
            result = template_str
        return NodeResult(success=True, data={}, output_text=result)


class MemoryWriteExecutor(NodeExecutor):
    """Writes the last message to flow memory."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        key = context.get_meta("memory_name", "memory")
        value = ""
        for msg in reversed(context.messages):
            if msg.content:
                value = msg.content
                break
        return NodeResult(
            success=True,
            data={"memory_updates": {key: value}},
            output_text=value,
        )


class MemoryReadExecutor(NodeExecutor):
    """Reads a value from flow memory."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        key = context.get_meta("memory_name", "memory")
        value = context.memory.get(key, "")
        return NodeResult(success=True, data={}, output_text=str(value))


class PassthroughExecutor(NodeExecutor):
    """Does nothing — passes through for nodes that don't need execution."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        text = ""
        for msg in reversed(context.messages):
            if msg.content:
                text = msg.content
                break
        return NodeResult(success=True, data={}, output_text=text)


class UserFormExecutor(NodeExecutor):
    """Auto-fills a user form with placeholder data."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        params = context.get_meta("parameters", [])
        form_data = {}
        for p in params:
            name = p.get("name", "field")
            form_data[name] = p.get("default", f"<{name}>")
        return NodeResult(
            success=True,
            data={"memory_updates": form_data},
            output_text=str(form_data),
        )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def _build_registry(
    provider_registry: ProviderRegistry, model: str, provider_name: str,
) -> SimpleNodeRegistry:
    """Create a node registry with executors for all common node types."""
    reg = SimpleNodeRegistry()
    llm = LLMExecutor(provider_registry, model, provider_name)
    decision = DecisionExecutor(provider_registry, model, provider_name)
    user = UserExecutor()
    static = StaticExecutor()
    var = VarExecutor()
    text = TextExecutor()
    mem_write = MemoryWriteExecutor()
    mem_read = MemoryReadExecutor()
    passthrough = PassthroughExecutor()
    user_form = UserFormExecutor()

    # LLM nodes
    reg.register(NodeType.INSTRUCTION.value, llm)

    reg.register(NodeType.SUMMARIZE.value, llm)
    reg.register(NodeType.AGENT.value, llm)
    reg.register(NodeType.INSTRUCTION_IMAGE_VISION.value, llm)

    # Decision nodes
    reg.register(NodeType.DECISION.value, decision)

    # User interaction
    reg.register(NodeType.USER.value, user)
    reg.register(NodeType.USER_DECISION.value, user)
    reg.register(NodeType.USER_FORM.value, user_form)
    reg.register(NodeType.USER_PROGRAM_FORM.value, user_form)

    # Data nodes
    reg.register(NodeType.STATIC.value, static)
    reg.register(NodeType.STATIC_DECISION.value, passthrough)
    reg.register(NodeType.TEXT.value, text)
    reg.register(NodeType.VAR.value, var)
    reg.register(NodeType.CODE.value, passthrough)

    # Memory nodes
    reg.register(NodeType.WRITE_MEMORY.value, mem_write)
    reg.register(NodeType.READ_MEMORY.value, mem_read)
    reg.register(NodeType.UPDATE_MEMORY.value, mem_write)
    reg.register(NodeType.FLOW_MEMORY.value, mem_read)
    reg.register(NodeType.USER_MEMORY.value, mem_read)

    # Merge / control
    reg.register(NodeType.STATIC_MERGE.value, passthrough)
    reg.register(NodeType.COMMENT.value, passthrough)
    reg.register(NodeType.BLANK.value, passthrough)
    reg.register(NodeType.SUB_ASSISTANT.value, passthrough)
    reg.register(NodeType.BREAK.value, passthrough)
    reg.register(NodeType.TEXT_TO_VARIABLE.value, var)
    if_exec = IfExecutor()
    reg.register(NodeType.IF.value, if_exec)
    reg.register(NodeType.SWITCH.value, passthrough)

    return reg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_graph(
    graph: AgentGraph,
    user_input: str = "Hello",
    provider: str | None = None,
    verbose: bool = True,
) -> FlowResult:
    """Build and execute a graph with real LLM calls.

    Args:
        graph: The built graph (from Graph.build() or the Graph itself).
        user_input: The initial user message.
        provider: Force a provider ("anthropic", "openai", etc.). Auto-detects if None.
        verbose: Print events as they happen.

    Returns:
        FlowResult with the final output.
    """
    # Resolve graph
    if hasattr(graph, "build"):
        agent_graph = graph.build()
    else:
        agent_graph = graph

    # Detect provider
    if provider:
        provider_name = provider
        model = _MODEL_MAP.get(provider, "gpt-4o")
    else:
        provider_name, model = _detect_provider()

    if not provider_name:
        print("No API key found. Set one of:")
        for p in _MODEL_MAP:
            print(f"  export {p.upper()}_API_KEY='...'")
        sys.exit(1)

    if verbose:
        print(f"Provider: {provider_name} ({model})")
        print(f"Input: {user_input!r}")
        print()

    # Set up provider registry
    provider_registry = _get_provider_registry(provider_name)

    # Set up node registry
    node_registry = _build_registry(provider_registry, model, provider_name)

    # Event handler — skip Start/End nodes to avoid duplicate echo
    _skip_types = {NodeType.START.value, NodeType.END.value}

    def on_event(event: FlowEvent) -> None:
        if not verbose:
            return
        if isinstance(event, NodeStarted):
            if event.node_type.value in _skip_types:
                return
            print(f"  [{event.node_type.value:15s}] {event.node_name}", flush=True)
        elif isinstance(event, NodeFinished):
            if event.node_id and any(
                n.id == event.node_id and n.type.value in _skip_types
                for n in agent_graph.nodes
            ):
                return
            output = event.result[:120] + "..." if len(event.result) > 120 else event.result
            if output:
                print(f"  {'':15s}   -> {output}", flush=True)
        elif isinstance(event, FlowError):
            print(f"  [ERROR] {event.error}", flush=True)

    # Run
    runner = FlowRunner(
        graph=agent_graph,
        node_registry=node_registry,
        store=InMemoryStore(),
        on_event=on_event,
    )

    result = runner.run(user_input)

    if verbose:
        print()
        if result.success:
            print(f"Final output:\n{result.final_output}")
        else:
            print(f"Flow failed: {result.error}")
        print(f"Duration: {result.duration_seconds:.2f}s")

    return result
