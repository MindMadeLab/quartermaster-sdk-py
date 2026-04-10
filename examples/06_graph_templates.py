"""Use built-in graph templates for common agent patterns.

The Templates class provides factory methods for frequently used graph
structures: chatbots, decision trees, RAG pipelines, tool-using agents,
and more.
"""

from __future__ import annotations

try:
    from quartermaster_graph.templates import Templates
    from quartermaster_graph.enums import NodeType
except ImportError:
    raise SystemExit("Install quartermaster-graph first:  pip install -e quartermaster-graph")


def _print_graph(name: str, graph) -> None:
    """Helper to print a graph summary."""
    print(f"\n{'=' * 60}")
    print(f"Template: {name}")
    print(f"  Nodes: {len(graph.nodes)}   Edges: {len(graph.edges)}   Version: {graph.version}")
    for node in graph.nodes:
        meta_keys = list(node.metadata.keys()) if node.metadata else []
        print(f"    {node.type.value:20s}  {node.name!r:30s}  meta={meta_keys}")


def main() -> None:
    # 1. Simple chat loop
    chat = Templates.simple_chat(
        name="My Chatbot",
        model="gpt-4o",
        system_instruction="You are a witty assistant.",
    )
    _print_graph("simple_chat", chat)

    # 2. Decision tree
    tree = Templates.decision_tree(
        name="Support Router",
        question="What type of issue?",
        options=["Billing", "Technical", "General"],
    )
    _print_graph("decision_tree", tree)

    # 3. RAG pipeline
    rag = Templates.rag_pipeline(retrieval_tool="semantic_search")
    _print_graph("rag_pipeline", rag)

    # 4. Multi-step processing
    multi = Templates.multi_step(steps=["Extract", "Transform", "Load"])
    _print_graph("multi_step", multi)

    # 5. Tool-using agent
    tool_agent = Templates.tool_using_agent(tools=["web_search", "calculator"])
    _print_graph("tool_using_agent", tool_agent)

    # 6. Multi-agent supervisor
    supervisor = Templates.multi_agent_supervisor(
        worker_names=["Researcher", "Writer", "Reviewer"],
    )
    _print_graph("multi_agent_supervisor", supervisor)

    # 7. Advanced RAG with reranking
    adv_rag = Templates.advanced_rag(
        retrieval_tool="vector_search",
        rerank_tool="cross_encoder",
    )
    _print_graph("advanced_rag", adv_rag)

    # 8. Parallel processing
    parallel = Templates.parallel_processing(branches=["Translate", "Summarize", "Extract"])
    _print_graph("parallel_processing", parallel)


if __name__ == "__main__":
    main()
