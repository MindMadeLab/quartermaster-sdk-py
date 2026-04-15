"""Semantic-change lint tool for the Quartermaster SDK.

This module catches pre-commit-time footguns where an SDK API changed
semantics between releases (e.g. the v0.3.0 ``.end()`` overload that was
reverted in v0.3.1).  It is NOT a general-purpose Python linter: the
rules are curated by SDK maintainers and each rule encodes ONE historical
or upcoming change plus its migration path.

Usage — Python API::

    from quartermaster_sdk.lint import check, list_rules, show_rule

    findings = check(["my_pkg/"], target_version="0.4.0")
    for f in findings:
        print(f.format())

    for rule in list_rules():
        print(rule.id, rule.summary)

    print(show_rule("QM001"))

Usage — CLI::

    python -m quartermaster_sdk.lint check [--target-version V] \\
        [--severity {warning,error}] PATH...
    python -m quartermaster_sdk.lint list-rules
    python -m quartermaster_sdk.lint show-rule QM001

Exit codes:
    0 — no issues at the requested severity.
    1 — at least one issue found.
    2 — invalid arguments / config error.

Output format::

    path/to/file.py:LINE: SEVERITY [QMxxx] message
      — see python -m quartermaster_sdk.lint show-rule QMxxx

Pre-commit hook example (NOT auto-installed — copy into
``.pre-commit-config.yaml`` if you want it)::

    - repo: local
      hooks:
        - id: quartermaster-lint
          name: Quartermaster API lint
          entry: python -m quartermaster_sdk.lint check --target-version 0.4.0
          language: system
          types: [python]
"""

from __future__ import annotations

from .checker import LintFinding, check_paths
from .rules import (
    RULES,
    SemanticChangeRule,
    all_rules,
    get_rule,
    rules_for_target,
)


def check(
    paths,
    *,
    target_version: str | None = None,
    min_severity: str = "warning",
) -> list[LintFinding]:
    """Scan ``paths`` and return findings (see :func:`check_paths`)."""

    return check_paths(paths, target_version=target_version, min_severity=min_severity)


def list_rules() -> tuple[SemanticChangeRule, ...]:
    """Return every curated rule (insertion order)."""

    return all_rules()


def show_rule(rule_id: str) -> str:
    """Return the ``advice`` field for a rule.

    Raises ``KeyError`` with a helpful list of valid IDs on typo.
    """

    return get_rule(rule_id).advice


__all__ = [
    "LintFinding",
    "RULES",
    "SemanticChangeRule",
    "check",
    "check_paths",
    "list_rules",
    "rules_for_target",
    "show_rule",
]
