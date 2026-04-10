"""Tests for versioning — create, fork, diff, semver helpers."""


import pytest

from qm_graph.enums import NodeType
from qm_graph.models import Agent, GraphEdge, GraphNode
from qm_graph.versioning import bump_major, bump_minor, bump_patch, create_version, diff, fork


class TestSemverHelpers:
    def test_bump_major(self):
        assert bump_major("1.2.3") == "2.0.0"
        assert bump_major("0.0.1") == "1.0.0"

    def test_bump_minor(self):
        assert bump_minor("1.2.3") == "1.3.0"
        assert bump_minor("0.0.0") == "0.1.0"

    def test_bump_patch(self):
        assert bump_patch("1.2.3") == "1.2.4"
        assert bump_patch("0.0.0") == "0.0.1"

    def test_invalid_semver(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            bump_major("abc")

    def test_invalid_semver_partial(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            bump_minor("1.2")


class TestCreateVersion:
    def test_basic(self, agent, start_node, end_node):
        edge = GraphEdge(source_id=start_node.id, target_id=end_node.id)
        version = create_version(
            agent=agent,
            version="1.0.0",
            nodes=[start_node, end_node],
            edges=[edge],
            start_node_id=start_node.id,
        )
        assert version.version == "1.0.0"
        assert version.agent_id == agent.id
        assert len(version.nodes) == 2
        assert len(version.edges) == 1

    def test_deep_copies_nodes(self, agent, start_node, end_node):
        edge = GraphEdge(source_id=start_node.id, target_id=end_node.id)
        version = create_version(
            agent=agent,
            version="1.0.0",
            nodes=[start_node, end_node],
            edges=[edge],
            start_node_id=start_node.id,
        )
        # Modifying original should not affect version
        start_node.name = "Modified"
        assert version.nodes[0].name != "Modified"

    def test_invalid_version_string(self, agent, start_node, end_node):
        edge = GraphEdge(source_id=start_node.id, target_id=end_node.id)
        with pytest.raises(ValueError):
            create_version(
                agent=agent,
                version="not-semver",
                nodes=[start_node, end_node],
                edges=[edge],
                start_node_id=start_node.id,
            )


class TestFork:
    def test_fork_creates_new_ids(self, simple_graph):
        new_agent = Agent(name="Forked Agent")
        forked = fork(simple_graph, new_agent)

        assert forked.agent_id == new_agent.id
        assert forked.forked_from == simple_graph.id
        assert forked.version == "0.1.0"

        # All node IDs should be different
        original_ids = {n.id for n in simple_graph.nodes}
        forked_ids = {n.id for n in forked.nodes}
        assert original_ids.isdisjoint(forked_ids)

    def test_fork_preserves_structure(self, simple_graph):
        new_agent = Agent(name="Forked")
        forked = fork(simple_graph, new_agent)

        assert len(forked.nodes) == len(simple_graph.nodes)
        assert len(forked.edges) == len(simple_graph.edges)

    def test_fork_edge_ids_remapped(self, simple_graph):
        new_agent = Agent(name="Forked")
        forked = fork(simple_graph, new_agent)

        forked_node_ids = {n.id for n in forked.nodes}
        for edge in forked.edges:
            assert edge.source_id in forked_node_ids
            assert edge.target_id in forked_node_ids


class TestDiff:
    def test_no_changes(self, simple_graph):
        d = diff(simple_graph, simple_graph)
        assert not d.has_changes

    def test_added_node(self, simple_graph, agent):
        import copy
        v2 = copy.deepcopy(simple_graph)
        v2.version = "0.2.0"
        new_node = GraphNode(type=NodeType.INSTRUCTION, name="New")
        v2.nodes.append(new_node)

        d = diff(simple_graph, v2)
        assert d.has_changes
        added = [nd for nd in d.node_diffs if nd.change == "added"]
        assert len(added) == 1
        assert added[0].new.name == "New"

    def test_removed_node(self, simple_graph, agent):
        import copy
        v2 = copy.deepcopy(simple_graph)
        v2.version = "0.2.0"
        v2.nodes.pop()  # remove last node

        d = diff(simple_graph, v2)
        removed = [nd for nd in d.node_diffs if nd.change == "removed"]
        assert len(removed) == 1

    def test_modified_node(self, simple_graph):
        import copy
        v2 = copy.deepcopy(simple_graph)
        v2.version = "0.2.0"
        v2.nodes[1].name = "Changed Name"

        d = diff(simple_graph, v2)
        modified = [nd for nd in d.node_diffs if nd.change == "modified"]
        assert len(modified) == 1

    def test_diff_versions_tracked(self, simple_graph):
        import copy
        v2 = copy.deepcopy(simple_graph)
        v2.version = "0.2.0"

        d = diff(simple_graph, v2)
        assert d.version_from == "0.1.0"
        assert d.version_to == "0.2.0"
