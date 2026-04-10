"""Pre-built graph templates for common agent patterns."""

from __future__ import annotations

from qm_graph.builder import GraphBuilder
from qm_graph.models import AgentVersion


class Templates:
    """Factory for common agent graph patterns."""

    @staticmethod
    def simple_chat(
        name: str = "Simple Chat Agent",
        model: str = "gpt-4o",
        system_instruction: str = "You are a helpful assistant.",
    ) -> AgentVersion:
        """Start -> Instruction -> User -> (loop back to Instruction) -> End.

        A basic chat loop agent.
        """
        builder = GraphBuilder(name)
        builder.start()
        builder.instruction(
            "Process",
            model=model,
            system_instruction=system_instruction,
        )
        builder.user("User Input")
        builder.end()
        return builder.build(validate=True)

    @staticmethod
    def decision_tree(
        name: str = "Decision Tree",
        question: str = "What should we do?",
        options: list[str] | None = None,
        model: str = "gpt-4o",
    ) -> AgentVersion:
        """Start -> Decision -> branches -> End.

        A decision tree with labeled branches.
        """
        if options is None:
            options = ["Yes", "No"]

        b = GraphBuilder(name)
        b.start()
        b.decision(question, options=options)
        for opt in options:
            b.on(opt).instruction(f"Handle: {opt}", model=model).end()
        return b.build(validate=True)

    @staticmethod
    def rag_pipeline(
        name: str = "RAG Pipeline",
        model: str = "gpt-4o",
        retrieval_tool: str = "vector_search",
    ) -> AgentVersion:
        """Start -> Tool (retrieve) -> Instruction (generate) -> End.

        Retrieval-Augmented Generation pipeline.
        """
        return (
            GraphBuilder(name)
            .start()
            .tool("Retrieve", tool_name=retrieval_tool)
            .instruction(
                "Generate",
                model=model,
                system_instruction="Answer based on retrieved context.",
            )
            .end()
            .build(validate=True)
        )

    @staticmethod
    def multi_step(
        name: str = "Multi-Step Agent",
        steps: list[str] | None = None,
        model: str = "gpt-4o",
    ) -> AgentVersion:
        """Start -> Instruction -> Code -> Instruction -> End.

        Multi-step processing pipeline.
        """
        if steps is None:
            steps = ["Analyze", "Process", "Summarize"]

        b = GraphBuilder(name)
        b.start()
        for step in steps:
            b.instruction(step, model=model)
        b.end()
        return b.build(validate=True)

    @staticmethod
    def parallel_processing(
        name: str = "Parallel Processing",
        branches: list[str] | None = None,
        model: str = "gpt-4o",
    ) -> AgentVersion:
        """Start -> Decision (fan-out) -> [branches] -> End.

        Parallel processing with multiple branches all ending.
        """
        if branches is None:
            branches = ["Branch A", "Branch B", "Branch C"]

        b = GraphBuilder(name)
        b.start()
        b.decision("Route", options=branches)
        for branch in branches:
            b.on(branch).instruction(branch, model=model).end()
        return b.build(validate=True)
