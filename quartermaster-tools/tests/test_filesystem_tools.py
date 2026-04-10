"""
Tests for the filesystem tools package.

At least 5 tests per tool (40+ total).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from quartermaster_tools.builtin.filesystem import (
    CopyFileTool,
    CreateDirectoryTool,
    DeleteFileTool,
    FileInfoTool,
    FindFilesTool,
    GrepTool,
    ListDirectoryTool,
    MoveFileTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp(tmp_path):
    """Return a tmp_path and create some test files inside it."""
    (tmp_path / "a.txt").write_text("hello world\nfoo bar\nbaz\n")
    (tmp_path / "b.py").write_text("import os\nprint('hi')\n")
    (tmp_path / ".hidden").write_text("secret\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("nested file\n")
    (sub / "d.py").write_text("# python\n")
    return tmp_path


# ===================================================================
# ListDirectoryTool
# ===================================================================

class TestListDirectoryTool:
    def test_basic_listing(self, tmp):
        tool = ListDirectoryTool()
        result = tool.run(path=str(tmp))
        assert result.success
        names = [e["name"] for e in result.data["entries"]]
        assert "a.txt" in names
        assert "sub" in names

    def test_hidden_excluded_by_default(self, tmp):
        tool = ListDirectoryTool()
        result = tool.run(path=str(tmp))
        names = [e["name"] for e in result.data["entries"]]
        assert ".hidden" not in names

    def test_hidden_included(self, tmp):
        tool = ListDirectoryTool()
        result = tool.run(path=str(tmp), include_hidden=True)
        names = [e["name"] for e in result.data["entries"]]
        assert ".hidden" in names

    def test_pattern_filter(self, tmp):
        tool = ListDirectoryTool()
        result = tool.run(path=str(tmp), pattern="*.py")
        names = [e["name"] for e in result.data["entries"]]
        assert "b.py" in names
        assert "a.txt" not in names

    def test_recursive(self, tmp):
        tool = ListDirectoryTool()
        result = tool.run(path=str(tmp), recursive=True)
        names = [e["name"] for e in result.data["entries"]]
        assert any("c.txt" in n for n in names)

    def test_not_a_directory(self, tmp):
        tool = ListDirectoryTool()
        result = tool.run(path=str(tmp / "a.txt"))
        assert not result.success
        assert "Not a directory" in result.error

    def test_blocked_path(self):
        tool = ListDirectoryTool()
        result = tool.run(path="/etc/passwd")
        assert not result.success
        assert "Access denied" in result.error

    def test_allowed_base_dir(self, tmp):
        tool = ListDirectoryTool(allowed_base_dir=str(tmp / "sub"))
        result = tool.run(path=str(tmp))
        assert not result.success
        assert "Access denied" in result.error


# ===================================================================
# FindFilesTool
# ===================================================================

class TestFindFilesTool:
    def test_glob_star(self, tmp):
        tool = FindFilesTool()
        result = tool.run(root_path=str(tmp), pattern="**/*.py")
        assert result.success
        assert result.data["count"] >= 2

    def test_simple_glob(self, tmp):
        tool = FindFilesTool()
        result = tool.run(root_path=str(tmp), pattern="*.txt")
        assert result.success
        assert any("a.txt" in f for f in result.data["files"])

    def test_name_pattern_regex(self, tmp):
        tool = FindFilesTool()
        result = tool.run(root_path=str(tmp), pattern="**/*", name_pattern=r"^[cd]\.")
        assert result.success
        names = [os.path.basename(f) for f in result.data["files"]]
        assert all(n.startswith(("c", "d")) for n in names)

    def test_invalid_regex(self, tmp):
        tool = FindFilesTool()
        result = tool.run(root_path=str(tmp), pattern="*", name_pattern="[invalid")
        assert not result.success
        assert "Invalid regex" in result.error

    def test_not_a_directory(self, tmp):
        tool = FindFilesTool()
        result = tool.run(root_path=str(tmp / "a.txt"), pattern="*")
        assert not result.success

    def test_blocked_path(self):
        tool = FindFilesTool()
        result = tool.run(root_path="/proc/self", pattern="*")
        assert not result.success

    def test_missing_params(self):
        tool = FindFilesTool()
        result = tool.run(root_path="", pattern="*")
        assert not result.success


# ===================================================================
# GrepTool
# ===================================================================

class TestGrepTool:
    def test_basic_search(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp / "a.txt"), pattern="hello")
        assert result.success
        assert result.data["total_matches"] == 1
        assert result.data["matches"][0]["line_number"] == 1

    def test_regex_search(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp / "a.txt"), pattern=r"fo+\s")
        assert result.success
        assert result.data["total_matches"] == 1

    def test_recursive_directory(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp), pattern="import")
        assert result.success
        assert result.data["total_matches"] >= 1

    def test_context_lines(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp / "a.txt"), pattern="foo", context_lines=1)
        assert result.success
        match = result.data["matches"][0]
        assert len(match["context"]) == 3  # line before, match, line after

    def test_no_match(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp / "a.txt"), pattern="zzzzz")
        assert result.success
        assert result.data["total_matches"] == 0

    def test_invalid_regex(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp / "a.txt"), pattern="[bad")
        assert not result.success
        assert "Invalid regex" in result.error

    def test_blocked_path(self):
        tool = GrepTool()
        result = tool.run(path="/etc/passwd", pattern="root")
        assert not result.success
        assert "Access denied" in result.error

    def test_non_recursive(self, tmp):
        tool = GrepTool()
        result = tool.run(path=str(tmp), pattern="nested", recursive=False)
        assert result.success
        assert result.data["total_matches"] == 0  # nested file is in sub/


# ===================================================================
# FileInfoTool
# ===================================================================

class TestFileInfoTool:
    def test_file_info(self, tmp):
        tool = FileInfoTool()
        result = tool.run(path=str(tmp / "a.txt"))
        assert result.success
        assert result.data["type"] == "file"
        assert result.data["size"] > 0
        assert "permissions" in result.data

    def test_directory_info(self, tmp):
        tool = FileInfoTool()
        result = tool.run(path=str(tmp / "sub"))
        assert result.success
        assert result.data["type"] == "directory"

    def test_mime_type(self, tmp):
        tool = FileInfoTool()
        result = tool.run(path=str(tmp / "b.py"))
        assert result.success
        # Python files are typically text/x-python
        assert result.data["mime_type"] is not None

    def test_not_found(self, tmp):
        tool = FileInfoTool()
        result = tool.run(path=str(tmp / "nonexistent"))
        assert not result.success
        assert "not found" in result.error.lower()

    def test_blocked_path(self):
        tool = FileInfoTool()
        result = tool.run(path="/etc/shadow")
        assert not result.success
        assert "Access denied" in result.error

    def test_missing_path_param(self):
        tool = FileInfoTool()
        result = tool.run()
        assert not result.success


# ===================================================================
# MoveFileTool
# ===================================================================

class TestMoveFileTool:
    def test_move_file(self, tmp):
        tool = MoveFileTool()
        src = str(tmp / "a.txt")
        dst = str(tmp / "moved.txt")
        result = tool.run(source=src, destination=dst)
        assert result.success
        assert os.path.exists(dst)
        assert not os.path.exists(src)

    def test_rename_file(self, tmp):
        tool = MoveFileTool()
        src = str(tmp / "b.py")
        dst = str(tmp / "renamed.py")
        result = tool.run(source=src, destination=dst)
        assert result.success

    def test_source_not_found(self, tmp):
        tool = MoveFileTool()
        result = tool.run(source=str(tmp / "nope"), destination=str(tmp / "x"))
        assert not result.success

    def test_missing_source_param(self, tmp):
        tool = MoveFileTool()
        result = tool.run(source="", destination=str(tmp / "x"))
        assert not result.success

    def test_blocked_destination(self, tmp):
        tool = MoveFileTool()
        result = tool.run(source=str(tmp / "a.txt"), destination="/etc/evil")
        assert not result.success
        assert "Access denied" in result.error

    def test_allowed_base_dir(self, tmp):
        tool = MoveFileTool(allowed_base_dir=str(tmp / "sub"))
        result = tool.run(source=str(tmp / "a.txt"), destination=str(tmp / "sub" / "a.txt"))
        assert not result.success  # source is outside base dir


# ===================================================================
# DeleteFileTool
# ===================================================================

class TestDeleteFileTool:
    def test_delete_file(self, tmp):
        tool = DeleteFileTool()
        target = str(tmp / "a.txt")
        result = tool.run(path=target, confirm=True)
        assert result.success
        assert not os.path.exists(target)

    def test_delete_without_confirm(self, tmp):
        tool = DeleteFileTool()
        result = tool.run(path=str(tmp / "a.txt"), confirm=False)
        assert not result.success
        assert "not confirmed" in result.error.lower()
        assert os.path.exists(str(tmp / "a.txt"))

    def test_delete_directory(self, tmp):
        tool = DeleteFileTool()
        result = tool.run(path=str(tmp / "sub"), confirm=True)
        assert result.success
        assert not os.path.exists(str(tmp / "sub"))

    def test_delete_not_found(self, tmp):
        tool = DeleteFileTool()
        result = tool.run(path=str(tmp / "nope"), confirm=True)
        assert not result.success

    def test_blocked_path(self):
        tool = DeleteFileTool()
        result = tool.run(path="/etc/passwd", confirm=True)
        assert not result.success
        assert "Access denied" in result.error

    def test_missing_path(self):
        tool = DeleteFileTool()
        result = tool.run(path="", confirm=True)
        assert not result.success


# ===================================================================
# CopyFileTool
# ===================================================================

class TestCopyFileTool:
    def test_copy_file(self, tmp):
        tool = CopyFileTool()
        src = str(tmp / "a.txt")
        dst = str(tmp / "copy_a.txt")
        result = tool.run(source=src, destination=dst)
        assert result.success
        assert os.path.exists(dst)
        assert os.path.exists(src)  # original still exists

    def test_copy_directory(self, tmp):
        tool = CopyFileTool()
        dst = str(tmp / "sub_copy")
        result = tool.run(source=str(tmp / "sub"), destination=dst)
        assert result.success
        assert os.path.isdir(dst)

    def test_source_not_found(self, tmp):
        tool = CopyFileTool()
        result = tool.run(source=str(tmp / "nope"), destination=str(tmp / "x"))
        assert not result.success

    def test_missing_params(self):
        tool = CopyFileTool()
        result = tool.run(source="", destination="foo")
        assert not result.success

    def test_blocked_source(self, tmp):
        tool = CopyFileTool()
        result = tool.run(source="/etc/passwd", destination=str(tmp / "x"))
        assert not result.success
        assert "Access denied" in result.error

    def test_blocked_destination(self, tmp):
        tool = CopyFileTool()
        result = tool.run(source=str(tmp / "a.txt"), destination="/etc/evil")
        assert not result.success

    def test_allowed_base_dir(self, tmp):
        tool = CopyFileTool(allowed_base_dir=str(tmp))
        src = str(tmp / "a.txt")
        dst = str(tmp / "copy2.txt")
        result = tool.run(source=src, destination=dst)
        assert result.success


# ===================================================================
# CreateDirectoryTool
# ===================================================================

class TestCreateDirectoryTool:
    def test_create_simple(self, tmp):
        tool = CreateDirectoryTool()
        target = str(tmp / "newdir")
        result = tool.run(path=target)
        assert result.success
        assert os.path.isdir(target)

    def test_create_nested(self, tmp):
        tool = CreateDirectoryTool()
        target = str(tmp / "a" / "b" / "c")
        result = tool.run(path=target, parents=True)
        assert result.success
        assert os.path.isdir(target)

    def test_no_parents_fails(self, tmp):
        tool = CreateDirectoryTool()
        target = str(tmp / "x" / "y" / "z")
        result = tool.run(path=target, parents=False)
        assert not result.success

    def test_existing_dir_with_parents(self, tmp):
        tool = CreateDirectoryTool()
        target = str(tmp / "sub")
        result = tool.run(path=target, parents=True)
        assert result.success  # exist_ok=True

    def test_existing_dir_without_parents(self, tmp):
        tool = CreateDirectoryTool()
        target = str(tmp / "sub")
        result = tool.run(path=target, parents=False)
        assert not result.success  # already exists

    def test_blocked_path(self):
        tool = CreateDirectoryTool()
        result = tool.run(path="/etc/evil")
        assert not result.success
        assert "Access denied" in result.error

    def test_missing_path(self):
        tool = CreateDirectoryTool()
        result = tool.run(path="")
        assert not result.success


# ===================================================================
# Security shared tests
# ===================================================================

class TestSecurityShared:
    """Cross-cutting security tests that apply to all tools."""

    def test_all_tools_block_proc(self, tmp):
        tools_with_path = [
            (ListDirectoryTool(), {"path": "/proc/self"}),
            (FindFilesTool(), {"root_path": "/proc/self", "pattern": "*"}),
            (GrepTool(), {"path": "/proc/version", "pattern": "x"}),
            (FileInfoTool(), {"path": "/proc/version"}),
            (DeleteFileTool(), {"path": "/proc/version", "confirm": True}),
            (CreateDirectoryTool(), {"path": "/proc/test"}),
        ]
        for tool, kwargs in tools_with_path:
            result = tool.run(**kwargs)
            assert not result.success, f"{tool.name()} should block /proc/"
            assert "Access denied" in result.error

    def test_all_tools_block_sys(self, tmp):
        tools_with_path = [
            (ListDirectoryTool(), {"path": "/sys/class"}),
            (FileInfoTool(), {"path": "/sys/class"}),
        ]
        for tool, kwargs in tools_with_path:
            result = tool.run(**kwargs)
            assert not result.success, f"{tool.name()} should block /sys/"

    def test_allowed_base_dir_validation(self, tmp):
        # All tools should raise ValueError if base dir is a file
        file_path = str(tmp / "a.txt")
        with pytest.raises(ValueError):
            ListDirectoryTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            FindFilesTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            GrepTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            FileInfoTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            MoveFileTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            DeleteFileTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            CopyFileTool(allowed_base_dir=file_path)
        with pytest.raises(ValueError):
            CreateDirectoryTool(allowed_base_dir=file_path)


# ===================================================================
# Tool interface conformance
# ===================================================================

class TestToolInterface:
    """Verify each tool implements the AbstractTool interface correctly."""

    @pytest.mark.parametrize("cls", [
        ListDirectoryTool, FindFilesTool, GrepTool, FileInfoTool,
        MoveFileTool, DeleteFileTool, CopyFileTool, CreateDirectoryTool,
    ])
    def test_interface(self, cls):
        tool = cls()
        assert isinstance(tool.name(), str)
        assert isinstance(tool.version(), str)
        assert len(tool.parameters()) > 0
        info = tool.info()
        assert info.name == tool.name()
        assert info.version == tool.version()
