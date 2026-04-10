"""Simple agent example -- build a minimal graph and run it.

This demonstrates the core Quartermaster workflow:
1. Define an agent graph using GraphBuilder
2. Create a node registry for execution
3. Run the graph with FlowRunner

Prerequisites:
    pip install qm-graph qm-engine
"""

from qm_graph import GraphBuilder
from qm_engine import FlowRunner
from qm_engine.nodes import SimpleNodeRegistry

# ------------------------------------------------------------------
# Step 1: Build the agent graph
# ------------------------------------------------------------------
# GraphBuilder provides a fluent API for constructing agent graphs.
# Every graph needs at least: start() -> one or more nodes -> end().
# The .instruction() method adds an LLM call node.

graph = (
    GraphBuilder("Simple Analyst", description="Analyzes user input")
    .start()
    .instruction(
        "Analyze input",
        model="gpt-4o",
        provider="openai",
        temperature=0.7,
        system_instruction="You are a helpful analyst. Summarize the input concisely.",
    )
    .end()
    .build()
)

# ------------------------------------------------------------------
# Step 2: Set up the node registry
# ------------------------------------------------------------------
# The node registry maps node type strings (e.g., "Instruction1") to
# executable node implementations. SimpleNodeRegistry is a basic
# in-memory registry provided by qm-engine.
#
# In a full setup, you would register node executors from qm-nodes:
#   from qm_nodes import NodeRegistry
#   node_registry = NodeRegistry()
#   node_registry.discover("qm_nodes.nodes")

node_registry = SimpleNodeRegistry()

# ------------------------------------------------------------------
# Step 3: Run the graph
# ------------------------------------------------------------------
# FlowRunner orchestrates the execution: it traverses the graph,
# executes each node, manages message passing, and collects results.

runner = FlowRunner(graph=graph, node_registry=node_registry)
result = runner.run("The global economy showed mixed signals in Q3 2025.")

# ------------------------------------------------------------------
# Step 4: Inspect the result
# ------------------------------------------------------------------
# FlowResult contains:
#   - success: whether the flow completed without errors
#   - final_output: the text output from the last End node
#   - flow_id: unique identifier for this execution
#   - duration_seconds: how long execution took
#   - node_results: per-node results keyed by node UUID

print(f"Success: {result.success}")
print(f"Output: {result.final_output}")
print(f"Duration: {result.duration_seconds:.2f}s")

if result.error:
    print(f"Error: {result.error}")
