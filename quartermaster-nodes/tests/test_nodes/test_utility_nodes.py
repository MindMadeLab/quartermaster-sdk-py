"""Tests for utility nodes."""

from tests.conftest import MockNodeContext, MockThought


class TestBlankNode:
    def test_think_does_nothing(self):
        from quartermaster_nodes.nodes.utility.blank import BlankNode

        ctx = MockNodeContext()
        BlankNode.think(ctx)
        assert len(ctx.handle.texts) == 0


class TestCommentNode:
    def test_think_does_nothing(self):
        from quartermaster_nodes.nodes.utility.comment import CommentNode

        ctx = MockNodeContext()
        CommentNode.think(ctx)
        assert len(ctx.handle.texts) == 0

    def test_config_no_edges(self):
        from quartermaster_nodes.nodes.utility.comment import CommentNode

        config = CommentNode.flow_config()
        assert not config.accepts_incoming_edges
        assert not config.accepts_outgoing_edges


class TestViewMetadataNode:
    def test_outputs_metadata(self):
        from quartermaster_nodes.nodes.utility.view_metadata import ViewMetadataNode

        ctx = MockNodeContext(thought=MockThought(metadata={"key": "value", "number": 42}))
        ViewMetadataNode.think(ctx)
        assert "key" in ctx.handle.last_text
        assert "value" in ctx.handle.last_text
        assert "42" in ctx.handle.last_text


class TestUseEnvironmentNode:
    def test_calls_activator(self):
        from quartermaster_nodes.nodes.utility.use_environment import UseEnvironmentNode

        activated = []
        ctx = MockNodeContext(
            node_metadata={
                "environment_id": "env-123",
                "_environment_activator": lambda eid, c: activated.append(eid),
            }
        )
        UseEnvironmentNode.think(ctx)
        assert activated == ["env-123"]


class TestUnselectEnvironmentNode:
    def test_calls_deactivator(self):
        from quartermaster_nodes.nodes.utility.unselect_environment import UnselectEnvironmentNode

        deactivated = []
        ctx = MockNodeContext(
            node_metadata={"_environment_deactivator": lambda c: deactivated.append(True)}
        )
        UnselectEnvironmentNode.think(ctx)
        assert len(deactivated) == 1


class TestUseFileNode:
    def test_calls_file_loader(self):
        from quartermaster_nodes.nodes.utility.use_file import UseFileNode

        loaded = []
        ctx = MockNodeContext(
            node_metadata={
                "file_id": "file-abc",
                "_file_loader": lambda fid, c: loaded.append(fid),
            }
        )
        UseFileNode.think(ctx)
        assert loaded == ["file-abc"]
