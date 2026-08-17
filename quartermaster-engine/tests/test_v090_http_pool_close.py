"""v0.9.0: FlowRunner acloses provider HTTP clients before asyncio.run returns.

Regression for GitHub #96 / #97. ``instruction()`` / ``Graph.run`` spin
``asyncio.run(executor)`` per node. If the provider's ``AsyncOpenAI`` /
httpx pool is still open when that loop shuts down, httpcore2 logs
``RuntimeError: generator didn't stop after athrow()``. The runner must
``await registry.aclose()`` inside the coroutine's ``finally`` — while
the loop is still running — not after ``asyncio.run()`` returns.
"""

from __future__ import annotations

from quartermaster_engine.nodes import SimpleNodeRegistry
from quartermaster_engine.runner.flow_runner import FlowRunner
from quartermaster_engine.types import NodeType, TraverseOut
from tests.conftest import EchoExecutor, make_edge, make_graph, make_node


class TrackingRegistry:
    """Duck-typed provider registry that records ``aclose()`` calls."""

    def __init__(self) -> None:
        self.aclose_calls = 0
        self.aclose_saw_running_loop = False

    async def aclose(self) -> None:
        import asyncio

        self.aclose_calls += 1
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self.aclose_saw_running_loop = loop is not None and loop.is_running()


def _echo_graph():
    start = make_node(NodeType.START, name="Start")
    instruction = make_node(NodeType.INSTRUCTION, name="Echo")
    end = make_node(NodeType.END, name="End", traverse_out=TraverseOut.SPAWN_NONE)
    graph = make_graph(
        [start, instruction, end],
        [make_edge(start, instruction), make_edge(instruction, end)],
        start,
    )
    nodes = SimpleNodeRegistry()
    nodes.register(NodeType.INSTRUCTION.value, EchoExecutor())
    return graph, nodes


class TestFlowRunnerAclosesProviders:
    def test_run_aclose_while_loop_still_running(self):
        graph, nodes = _echo_graph()
        tracking = TrackingRegistry()
        runner = FlowRunner(
            graph=graph,
            node_registry=nodes,
            provider_registry=tracking,
        )
        result = runner.run("Hello")

        assert result.success
        assert tracking.aclose_calls >= 1
        assert tracking.aclose_saw_running_loop, (
            "aclose() must run inside asyncio.run()'s coroutine, not after "
            "the loop has closed — otherwise httpx/httpcore2 cannot drain "
            "PoolByteStream and stderr prints GeneratorExit / athrow()"
        )

    def test_run_without_provider_registry_still_succeeds(self):
        """Runners that only have a node_registry must not crash on aclose."""
        graph, nodes = _echo_graph()
        runner = FlowRunner(graph=graph, node_registry=nodes)
        result = runner.run("Hello")
        assert result.success
        assert "Hello" in result.final_output
