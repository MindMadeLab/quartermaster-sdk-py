"""Lock in the v0.6.0 removal of lint rules QM002, QM003, QM004.

Each of those rules documented an already-shipped breaking change:

* QM002 — ``.start()`` was redundant since v0.2.0; Graph auto-creates
  the Start node.  Fully shipped.
* QM003 — ``.end(stop=True)`` was a v0.3.0-only kwarg removed in
  v0.3.1.  Fully shipped.
* QM004 — ``[IMAGE_BASE64::...]`` shim strings were replaced by the
  ``image=`` kwarg in v0.3.0.  Fully shipped.

In v0.6.0 the rules themselves are dropped from the curated database.
These tests confirm:

1. Each removed rule id is absent from ``list_rules()``.
2. ``check()`` on source that previously triggered the rule does NOT
   emit the removed finding.
3. The surviving rules (QM001, QM005) still work.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# Mirror the parent-package isolation trick from test_v040_lint.py so
# the tests run even if quartermaster_sdk's top-level __init__ fails to
# import in isolation.
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


REMOVED_IDS = ("QM002", "QM003", "QM004")


@pytest.mark.parametrize("rule_id", REMOVED_IDS)
def test_removed_rule_not_in_list_rules(rule_id: str) -> None:
    """Each removed rule id must be absent from the curated list."""

    ids = {r.id for r in list_rules()}
    assert rule_id not in ids


@pytest.mark.parametrize("rule_id", REMOVED_IDS)
def test_show_rule_for_removed_id_raises(rule_id: str) -> None:
    """``show_rule()`` on a removed id must raise KeyError, not return stale text."""

    with pytest.raises(KeyError):
        show_rule(rule_id)


def test_start_call_no_longer_flags_qm002(tmp_path: Path) -> None:
    """A ``.start()`` call used to fire QM002.  It must now scan clean."""

    (tmp_path / "uses_start.py").write_text(
        "from quartermaster_sdk import Graph\n"
        'g = Graph("x").start().user().agent().build()\n',
        encoding="utf-8",
    )

    findings = check([tmp_path])
    assert not any(f.rule.id == "QM002" for f in findings)


def test_end_stop_true_no_longer_flags_qm003(tmp_path: Path) -> None:
    """``.end(stop=True)`` used to fire QM003 at error severity.  Gone now."""

    (tmp_path / "uses_end_stop.py").write_text(
        "def f(g):\n    return g.end(stop=True)\n",
        encoding="utf-8",
    )

    findings = check([tmp_path])
    assert not any(f.rule.id == "QM003" for f in findings)


def test_image_base64_shim_no_longer_flags_qm004(tmp_path: Path) -> None:
    """``[IMAGE_BASE64::...]`` used to fire QM004.  Gone now."""

    (tmp_path / "uses_image_shim.py").write_text(
        'prompt = f"[IMAGE_BASE64::{b64}] describe this"\n',
        encoding="utf-8",
    )

    findings = check([tmp_path])
    assert not any(f.rule.id == "QM004" for f in findings)


def test_qm001_still_fires_at_target_0_3_0(tmp_path: Path) -> None:
    """Sanity: surviving QM001 still fires against a bare ``.end()`` at v0.3.0."""

    (tmp_path / "bare_end.py").write_text(
        "def f(g):\n    return g.end()\n",
        encoding="utf-8",
    )

    findings = check([tmp_path], target_version="0.3.0")
    assert any(f.rule.id == "QM001" for f in findings)


def test_qm005_still_fires_on_direct_flowrunner_import(tmp_path: Path) -> None:
    """Sanity: surviving QM005 still fires on direct FlowRunner use."""

    (tmp_path / "uses_flowrunner.py").write_text(
        "from quartermaster_engine import FlowRunner\n"
        "def run(g): return FlowRunner().run(g)\n",
        encoding="utf-8",
    )

    findings = check([tmp_path])
    assert any(f.rule.id == "QM005" for f in findings)
