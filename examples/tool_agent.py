"""Tool-calling agent example -- an agent that uses tools.

This demonstrates how to:
1. Define custom tools using qm-tools
2. Register tools in a ToolRegistry
3. Build a graph with a tool invocation node
4. Wire everything together with FlowRunner

Prerequisites:
    pip install qm-graph qm-engine qm-tools
"""

from typing import Any

from qm_tools import (
    AbstractTool,
    ToolDescriptor,
    ToolParameter,
    ToolRegistry,
    ToolResult,
)
from qm_graph import GraphBuilder
from qm_engine import FlowRunner
from qm_engine.nodes import SimpleNodeRegistry


# ------------------------------------------------------------------
# Step 1: Define a custom tool
# ------------------------------------------------------------------
# Tools extend AbstractTool and implement: name(), version(),
# parameters(), info(), and run(). The tool registry can export
# tool schemas in OpenAI, Anthropic, or MCP format for LLM
# function calling.

class WordCountTool(AbstractTool):
    """A simple tool that counts words in a text."""

    def name(self) -> str:
        return "word_count"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="string",
                description="The text to count words in",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="word_count",
            short_description="Count the number of words in a text",
            long_description="Splits text by whitespace and returns the word count.",
            version="1.0.0",
        )

    def run(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get("text", "")
        word_count = len(text.split())
        return ToolResult(
            success=True,
            data={"word_count": word_count, "text_length": len(text)},
        )


class TextReverseTool(AbstractTool):
    """A tool that reverses text."""

    def name(self) -> str:
        return "text_reverse"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="string",
                description="The text to reverse",
                required=True,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="text_reverse",
            short_description="Reverse a string of text",
            long_description="Returns the input text with characters in reverse order.",
            version="1.0.0",
        )

    def run(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get("text", "")
        return ToolResult(
            success=True,
            data={"reversed": text[::-1]},
        )


# ------------------------------------------------------------------
# Step 2: Register tools
# ------------------------------------------------------------------
# ToolRegistry manages tool instances. It supports version-aware
# lookup and can export schemas for LLM function calling.

tool_registry = ToolRegistry()
tool_registry.register(WordCountTool())
tool_registry.register(TextReverseTool())

# Export tool schemas in different formats for LLM consumption
openai_tools = tool_registry.to_openai_tools()
anthropic_tools = tool_registry.to_anthropic_tools()
mcp_tools = tool_registry.to_mcp_tools()

print("Registered tools:")
for descriptor in tool_registry.list_tools():
    print(f"  - {descriptor.name} v{descriptor.version}: {descriptor.short_description}")

# ------------------------------------------------------------------
# Step 3: Build a graph that uses tools
# ------------------------------------------------------------------
# The .tool() method adds a tool invocation node to the graph.
# The .instruction() node can use tool results from previous nodes.

graph = (
    GraphBuilder("Tool Agent", description="Agent that uses tools to process text")
    .start()

    # Call the word_count tool on the user's input
    .tool("Count words", tool_name="word_count")

    # Use an LLM to interpret the tool results
    .instruction(
        "Summarize results",
        model="gpt-4o",
        provider="openai",
        system_instruction=(
            "You received the results of a word count analysis. "
            "Summarize the findings for the user in a friendly way."
        ),
    )

    .end()
    .build()
)

# ------------------------------------------------------------------
# Step 4: Run the agent
# ------------------------------------------------------------------

node_registry = SimpleNodeRegistry()
runner = FlowRunner(graph=graph, node_registry=node_registry)

result = runner.run("Quartermaster is an open-source AI agent orchestration framework.")

print(f"\nSuccess: {result.success}")
print(f"Output: {result.final_output}")

# ------------------------------------------------------------------
# Step 5: Use tools directly (without the graph engine)
# ------------------------------------------------------------------
# Tools can also be used standalone, outside of any agent graph.

print("\n--- Direct tool usage ---")

word_tool = tool_registry.get("word_count")
direct_result = word_tool.run(text="Hello world from Quartermaster")
print(f"Word count: {direct_result.data['word_count']}")

# safe_run validates parameters before execution
invalid_result = word_tool.safe_run()  # Missing required 'text' parameter
print(f"Validation result: success={invalid_result.success}, error={invalid_result.error}")
