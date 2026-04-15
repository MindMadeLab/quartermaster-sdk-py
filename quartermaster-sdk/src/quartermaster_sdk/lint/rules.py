"""Curated rule database for the Quartermaster SDK semantic-change linter.

Each rule encodes ONE historical or upcoming SDK API change plus its
migration path.  This is intentionally NOT a general-purpose linter: we
only ship rules that cover real churn maintainers have caused.

v0.4.0 ships regex-only matching to keep the impl small.  AST-based
patterns are a v0.4.1+ enhancement — see ``pattern_kind`` and the TODO
in ``checker.py`` for the extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["warning", "error"]
PatternKind = Literal["regex", "ast"]


@dataclass(frozen=True)
class SemanticChangeRule:
    """One curated SDK-API-change pattern.

    Attributes:
        id: Stable rule identifier (e.g. ``"QM001"``).
        summary: One-line human-readable description shown in listings.
        introduced_in: SDK version where this concern first applied.
        reverted_in: SDK version where the concern was removed
            (``None`` means the rule is still active).
        severity: ``"warning"`` or ``"error"``.
        pattern: Regex source (or AST expression when
            ``pattern_kind == "ast"``).
        pattern_kind: ``"regex"`` for v0.4.0; ``"ast"`` reserved for a
            future release (see TODO in checker.py).
        advice: Multiline migration guidance.  Rendered by ``show-rule``.
    """

    id: str
    summary: str
    introduced_in: str
    pattern: str
    advice: str
    reverted_in: str | None = None
    severity: Severity = "warning"
    pattern_kind: PatternKind = "regex"


# The curated database.  Keep ordering stable — ``list-rules`` prints in
# insertion order so IDs read like a chronology.
RULES: tuple[SemanticChangeRule, ...] = (
    SemanticChangeRule(
        id="QM001",
        summary=(
            ".end() was briefly overloaded as a loop-back in 0.3.0; reverted in 0.3.1."
        ),
        introduced_in="0.3.0",
        reverted_in="0.3.1",
        severity="warning",
        # Matches a bare ``.end()`` call.  Historical only: this rule
        # fires solely when --target-version is 0.3.0.
        pattern=r"\.end\s*\(\s*\)",
        advice=(
            "QM001 — .end() semantics changed briefly in 0.3.0\n"
            "\n"
            "v0.3.0 silently re-purposed ``.end()`` as a loop-back.  "
            "v0.3.1 reverted this before any production adoption.\n"
            "\n"
            "If you must target 0.3.0 exactly, use ``.back()`` for "
            "explicit loop-back or ``.end(stop=True)`` for a terminal "
            "stop.  For 0.3.1+, ``.end()`` returns to stop semantics and "
            "this rule no longer fires."
        ),
    ),
    SemanticChangeRule(
        id="QM002",
        summary=(".start() is deprecated since 0.2.0 — Graph auto-adds a Start node."),
        introduced_in="0.2.0",
        reverted_in=None,
        severity="warning",
        # Matches ``.start()`` — no args, chained from the Graph builder.
        pattern=r"\.start\s*\(\s*\)",
        advice=(
            "QM002 — .start() is redundant\n"
            "\n"
            "Since v0.2.0, ``Graph(...)`` auto-adds a Start node.  "
            "Calling ``.start()`` explicitly is a no-op that still lands "
            "in reviewers' blind spots as if it did something.\n"
            "\n"
            "Drop the ``.start()`` call entirely:\n"
            "\n"
            "    # before\n"
            '    Graph("x").start().user().agent().build()\n'
            "\n"
            "    # after\n"
            '    Graph("x").user().agent().build()'
        ),
    ),
    SemanticChangeRule(
        id="QM003",
        summary=(".end(stop=True) was removed in 0.3.1 — the kwarg no longer exists."),
        introduced_in="0.3.0",
        reverted_in="0.3.1",
        severity="error",
        # Match ``.end(`` ... ``stop`` ... ``=`` ... ``True`` ... ``)``.
        # Regex is deliberately permissive about whitespace and other
        # kwargs; the AST-based rewrite (v0.4.1+) will tighten this.
        pattern=r"\.end\s*\([^)]*\bstop\s*=\s*True[^)]*\)",
        advice=(
            "QM003 — .end(stop=True) no longer exists\n"
            "\n"
            "The ``stop`` kwarg on ``.end()`` was introduced in v0.3.0 "
            "and removed in v0.3.1.  ``.end()`` has reverted to always-"
            "stop semantics, so the kwarg is redundant and will raise "
            "``TypeError`` at graph-build time.\n"
            "\n"
            "Drop the kwarg:\n"
            "\n"
            "    # before\n"
            "    graph.end(stop=True)\n"
            "\n"
            "    # after\n"
            "    graph.end()"
        ),
    ),
    SemanticChangeRule(
        id="QM004",
        summary=(
            "[IMAGE_BASE64::...] shim strings are legacy — use image= "
            "kwarg since 0.3.0."
        ),
        introduced_in="0.3.0",
        reverted_in=None,
        severity="warning",
        # Match the legacy inline marker the pre-vision-kwarg codepath
        # used to stitch images into the user prompt.
        pattern=r"\[IMAGE_BASE64::[^\]]*\]",
        advice=(
            "QM004 — [IMAGE_BASE64::...] shim strings are legacy\n"
            "\n"
            "Before v0.3.0, users concatenated base64-encoded images "
            "into the user prompt via ``[IMAGE_BASE64::...]`` markers.  "
            "v0.3.0 introduced a first-class ``image=`` kwarg on "
            "``qm.run`` / ``qm.arun`` / ``qm.instruction`` that is "
            "strongly typed, validated, and routed correctly across "
            "providers.\n"
            "\n"
            "Replace:\n"
            "\n"
            "    run(graph, f'[IMAGE_BASE64::{b64}] describe this')\n"
            "\n"
            "with:\n"
            "\n"
            "    run(graph, 'describe this', image=b64)"
        ),
    ),
    SemanticChangeRule(
        id="QM005",
        summary=(
            "Direct quartermaster_engine.FlowRunner import is low-level "
            "— use qm.run / qm.arun since 0.2.1."
        ),
        introduced_in="0.2.1",
        reverted_in=None,
        severity="warning",
        # Catch ``from quartermaster_engine import ... FlowRunner`` and
        # ``import quartermaster_engine`` + ``quartermaster_engine.FlowRunner``.
        pattern=(
            r"(?:from\s+quartermaster_engine\s+import[^\n]*\bFlowRunner\b"
            r"|\bquartermaster_engine\.FlowRunner\b)"
        ),
        advice=(
            "QM005 — FlowRunner is the low-level API\n"
            "\n"
            "Since v0.2.1 the recommended entry points are ``qm.run`` "
            "(sync) and ``qm.arun`` (async).  Constructing ``FlowRunner`` "
            "directly pins callers to the legacy execution surface and "
            "misses features like typed chunks, tracing, and cancellation.\n"
            "\n"
            "Replace:\n"
            "\n"
            "    from quartermaster_engine import FlowRunner\n"
            "    FlowRunner(...).run(graph, user_input)\n"
            "\n"
            "with:\n"
            "\n"
            "    from quartermaster_sdk import run\n"
            "    run(graph, user_input)"
        ),
    ),
)

# Public lookup helper.
_RULES_BY_ID: dict[str, SemanticChangeRule] = {r.id: r for r in RULES}


def all_rules() -> tuple[SemanticChangeRule, ...]:
    """Return the curated rule list (immutable snapshot)."""

    return RULES


def get_rule(rule_id: str) -> SemanticChangeRule:
    """Return the rule with the given id or raise ``KeyError`` listing valid ids.

    The ``KeyError`` message includes the full list of known rule IDs so a
    typo surfaces with an actionable hint.
    """

    try:
        return _RULES_BY_ID[rule_id]
    except KeyError as exc:
        valid = ", ".join(sorted(_RULES_BY_ID))
        raise KeyError(
            f"Unknown rule id {rule_id!r}. Valid rule ids: {valid}."
        ) from exc


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple for comparisons.

    We only need ordering for the small SDK history (0.1.x → 0.4.x), so
    a plain tuple-of-ints works — no need to pull in ``packaging``.
    """

    return tuple(int(p) for p in v.split("."))


def rules_for_target(target_version: str | None) -> tuple[SemanticChangeRule, ...]:
    """Return the subset of rules that apply for the given target version.

    Selection rules:
    * If ``target_version`` is ``None``, return every rule.
    * A rule with ``reverted_in == target_version`` — the change was
      actively undone at this version — is skipped.
    * A rule with ``introduced_in <= target_version < reverted_in`` fires.
    * A rule with no ``reverted_in`` fires whenever the target is at or
      above ``introduced_in``.
    """

    if target_version is None:
        return RULES

    target = _version_tuple(target_version)
    out: list[SemanticChangeRule] = []
    for rule in RULES:
        intro = _version_tuple(rule.introduced_in)
        if target < intro:
            continue
        if rule.reverted_in is not None:
            reverted = _version_tuple(rule.reverted_in)
            if target >= reverted:
                continue
        out.append(rule)
    return tuple(out)


# Unused import sink — keeps ``field`` available for consumers who want
# to subclass ``SemanticChangeRule`` without re-importing dataclasses.
_ = field
