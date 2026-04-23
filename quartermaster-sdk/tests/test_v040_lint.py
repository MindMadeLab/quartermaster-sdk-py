"""Tests for the v0.4.0 semantic-change lint tool.

Covers the ``quartermaster_sdk.lint`` public surface:

* ``check()`` matches rules against real Python files.
* ``list_rules()`` / ``show_rule()`` behave as documented.
* The ``python -m quartermaster_sdk.lint`` CLI returns the right exit
  codes (0 clean, 1 findings, 2 bad input).
* ``--target-version`` filters historical-only rules.
"""

from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest


# ── parent-package isolation ───────────────────────────────────────────
#
# The ``quartermaster_sdk`` package __init__ re-exports names from sibling
# v0.4.0 work (timeouts / cancel / inline tools) that may land on this
# branch before my commit. To keep the lint tests self-contained — and
# to pass even while sibling v0.4.0 threads are in flight — we stub the
# top-level package if (and only if) it fails to import normally. The
# lint module has zero runtime dependencies on its parent, so loading it
# under a stub parent is behaviour-equivalent to loading it normally.
def _ensure_lint_importable() -> None:
    try:
        import quartermaster_sdk  # noqa: F401 — probing real import

        return
    except ImportError:
        pass
    here = Path(__file__).resolve().parent.parent / "src" / "quartermaster_sdk"
    pkg = types.ModuleType("quartermaster_sdk")
    pkg.__path__ = [str(here)]
    sys.modules["quartermaster_sdk"] = pkg


_ensure_lint_importable()

from quartermaster_sdk.lint import check, list_rules, show_rule  # noqa: E402
from quartermaster_sdk.lint.__main__ import main as lint_main  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────


def _write(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body, encoding="utf-8")
    return p


# ── rule-matching tests ────────────────────────────────────────────────


def test_check_clean_file_zero_warnings(tmp_path: Path) -> None:
    """File with no offending patterns returns an empty list."""

    _write(
        tmp_path,
        "clean.py",
        "from quartermaster_sdk import Graph, run\n"
        'g = Graph("x").user().agent().build()\n'
        'result = run(g, "hello")\n',
    )

    findings = check([tmp_path])

    assert findings == []


# ── list/show tests ────────────────────────────────────────────────────


def test_list_rules_emits_table() -> None:
    """``list_rules()`` returns every curated rule, each with id + summary."""

    rules = list_rules()
    ids = {r.id for r in rules}
    assert {"QM001", "QM005"} <= ids
    # QM002, QM003, QM004 were removed in v0.6.0 — they documented
    # breaking changes that have since fully shipped.
    assert {"QM002", "QM003", "QM004"}.isdisjoint(ids)
    for rule in rules:
        assert rule.id.startswith("QM")
        assert rule.summary  # non-empty
        assert rule.advice  # non-empty


def test_show_rule_returns_advice() -> None:
    """``show_rule("QM001")`` returns the migration advice body."""

    advice = show_rule("QM001")
    assert "QM001" in advice
    assert ".end()" in advice  # QM001 is the .end() overload rule


def test_unknown_rule_id_show_returns_helpful_error() -> None:
    """Typos surface the list of valid IDs, not just a raw KeyError."""

    with pytest.raises(KeyError) as excinfo:
        show_rule("QM999")
    msg = str(excinfo.value)
    assert "QM999" in msg
    assert "QM001" in msg  # valid IDs listed so the user can fix the typo


# ── target-version filter ──────────────────────────────────────────────


def test_target_version_filters_rules(tmp_path: Path) -> None:
    """QM001 is 0.3.0-only (reverted in 0.3.1). --target-version 0.4.0 skips it."""

    # A file that would fire QM001 if the rule were active: a bare .end().
    _write(tmp_path, "end.py", "def f(g):\n    return g.end()\n")

    # target-version 0.3.0: QM001 applies.
    findings_030 = check([tmp_path], target_version="0.3.0")
    assert any(f.rule.id == "QM001" for f in findings_030)

    # target-version 0.4.0: QM001 must be skipped.
    findings_040 = check([tmp_path], target_version="0.4.0")
    assert not any(f.rule.id == "QM001" for f in findings_040)


# ── CLI exit codes ────────────────────────────────────────────────────


def test_cli_main_exit_codes(tmp_path: Path) -> None:
    """Run the CLI programmatically against clean / warn fixtures.

    v0.6.0 note: QM002/QM003/QM004 were removed, and QM003 was the only
    error-severity rule.  The remaining rules (QM001, QM005) are all
    ``warning`` severity, so this test only exercises warning-level
    fixtures now.
    """

    clean = _write(tmp_path, "clean.py", "x = 1\n")
    warn = _write(
        tmp_path,
        "warn.py",
        "from quartermaster_engine import FlowRunner\n"
        "def run(g): return FlowRunner().run(g)\n",
    )

    # 0: no findings.
    with redirect_stdout(io.StringIO()):
        assert lint_main(["check", str(clean)]) == 0

    # 1: warning-level finding reported.
    out = io.StringIO()
    with redirect_stdout(out):
        rc = lint_main(["check", str(warn)])
    assert rc == 1
    assert "QM005" in out.getvalue()

    # 0: warning-only file reports nothing under --severity=error.
    with redirect_stdout(io.StringIO()):
        assert lint_main(["check", "--severity", "error", str(warn)]) == 0

    # 2: invalid rule id for show-rule.
    err_buf = io.StringIO()
    with redirect_stderr(err_buf):
        assert lint_main(["show-rule", "QM999"]) == 2
    assert "QM999" in err_buf.getvalue()
