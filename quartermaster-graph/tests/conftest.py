"""Shared fixtures for quartermaster-graph tests."""

from __future__ import annotations

import pytest

from quartermaster_graph.enums import (
    NodeType,
    TraverseOut,
)
from quartermaster_graph.models import Agent, GraphSpec, GraphEdge, GraphNode


@pytest.fixture
def agent() -> Agent:
    """A simple test agent."""
    return Agent(name="Test Agent", description="For testing")


@pytest.fixture
def start_node() -> GraphNode:
    """A Start node."""
    return GraphNode(type=NodeType.START, name="Start")


@pytest.fixture
def end_node() -> GraphNode:
    """An End node."""
    return GraphNode(type=NodeType.END, name="End")


@pytest.fixture
def instruction_node() -> GraphNode:
    """An Instruction node with metadata."""
    return GraphNode(
        type=NodeType.INSTRUCTION,
        name="Process",
        metadata={
            "llm_system_instruction": "You are helpful",
            "llm_model": "gpt-4o",
            "llm_provider": "openai",
            "llm_temperature": 0.7,
        },
    )


@pytest.fixture
def simple_graph(
    start_node: GraphNode,
    instruction_node: GraphNode,
    end_node: GraphNode,
    agent: Agent,
) -> GraphSpec:
    """A minimal valid graph: Start -> Instruction -> End."""
    edge1 = GraphEdge(source_id=start_node.id, target_id=instruction_node.id)
    edge2 = GraphEdge(source_id=instruction_node.id, target_id=end_node.id)
    return GraphSpec(
        agent_id=agent.id,
        start_node_id=start_node.id,
        nodes=[start_node, instruction_node, end_node],
        edges=[edge1, edge2],
    )


@pytest.fixture
def decision_graph(agent: Agent) -> GraphSpec:
    """A graph with a decision: Start -> Decision -> (Yes->End1, No->End2)."""
    start = GraphNode(type=NodeType.START, name="Start")
    decision = GraphNode(
        type=NodeType.DECISION,
        name="Choose",
        traverse_out=TraverseOut.SPAWN_PICKED,
    )
    yes_node = GraphNode(type=NodeType.INSTRUCTION, name="Yes Path")
    no_node = GraphNode(type=NodeType.INSTRUCTION, name="No Path")
    end1 = GraphNode(type=NodeType.END, name="End1")
    end2 = GraphNode(type=NodeType.END, name="End2")

    edges = [
        GraphEdge(source_id=start.id, target_id=decision.id),
        GraphEdge(source_id=decision.id, target_id=yes_node.id, label="Yes"),
        GraphEdge(source_id=decision.id, target_id=no_node.id, label="No"),
        GraphEdge(source_id=yes_node.id, target_id=end1.id),
        GraphEdge(source_id=no_node.id, target_id=end2.id),
    ]

    return GraphSpec(
        agent_id=agent.id,
        start_node_id=start.id,
        nodes=[start, decision, yes_node, no_node, end1, end2],
        edges=edges,
    )
