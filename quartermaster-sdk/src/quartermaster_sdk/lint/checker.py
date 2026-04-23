"""File scanner for the Quartermaster SDK semantic-change linter.

The scanner walks each target path, reads ``.py`` files line-by-line, and
runs the regex patterns from :mod:`quartermaster_sdk.lint.rules` against
every line.  Matches are reported as :class:`LintFinding` instances.

v0.4.0 is deliberately regex-only.  ``SemanticChangeRule.pattern_kind``
allows room for AST-based matching in v0.4.1+ — see the TODO in
:func:`_match_rule` for the dispatch extension point.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .rules import SemanticChangeRule, rules_for_target


@dataclass(frozen=True)
class LintFinding:
    """One rule match at one location."""

    path: Path
    line: int
    rule: SemanticChangeRule
    snippet: str

    def format(self) -> str:
        """Render the finding in the canonical CLI output shape."""

        return (
            f"{self.path}:{self.line}: {self.rule.severity.upper()} "
            f"[{self.rule.id}] {self.rule.summary} "
            f"— see python -m quartermaster_sdk.lint show-rule {self.rule.id}"
        )


def _iter_python_files(targets: Iterable[Path]) -> Iterator[Path]:
    """Yield every ``.py`` file reachable from ``targets``.

    Files are yielded directly; directories are walked recursively.
    Hidden directories (``.git``, ``.venv``, ``__pycache__``) are pruned
    so we don't lint vendored or build artifacts.
    """

    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
    }

    for target in targets:
        if target.is_file():
            if target.suffix == ".py":
                yield target
            continue
        if not target.is_dir():
            # Silently skip non-existent / special files; the CLI layer
            # surfaces missing-path errors before we get here.
            continue
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                if name.endswith(".py"):
                    yield Path(dirpath) / name


def _match_rule(rule: SemanticChangeRule, line: str) -> bool:
    """Apply a single rule to a single line.

    v0.4.0: regex only.

    TODO(v0.4.1+): when ``rule.pattern_kind == "ast"``, dispatch to an
    AST-walker that evaluates a structured pattern against the parsed
    module.  The regex-only path handles the current rule set fine,
    but more structural patterns are a good candidate for an AST
    rewrite once the rule set grows.
    """

    if rule.pattern_kind == "regex":
        return re.search(rule.pattern, line) is not None
    # Unknown kinds are treated as no-match rather than crashing the
    # whole run — forward compatibility for an older SDK install lint
    # against a newer rule file.
    return False


_TRIPLE_QUOTE_RE = re.compile(r'"""|\'\'\'')


def _scan_file(
    path: Path, rules: Iterable[SemanticChangeRule]
) -> Iterator[LintFinding]:
    """Yield every finding in ``path`` for the given rules.

    Lines inside triple-quoted strings (docstrings, doctests) and
    comment-only lines are skipped.  This keeps the linter from nagging
    on migration notes embedded in module / function documentation — a
    real source of noise against the examples/ tree.

    The triple-quote tracker is a coarse toggle: it counts triple-
    double-quote and triple-single-quote occurrences per line and flips
    state on each one.  It does not distinguish the two quote styles,
    so an odd pathology like a mix of the two on one line could mis-
    bucket it — the tighter AST rewrite (v0.4.1+) handles that cleanly.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Unreadable file: skip.  The CLI keeps going so one broken
        # file doesn't derail a whole tree scan.
        return

    in_docstring = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()

        # Track triple-quote state before deciding whether to lint.
        # Count triple-quotes on the line; if odd, we flip state.
        triple_count = len(_TRIPLE_QUOTE_RE.findall(line))
        line_started_in_docstring = in_docstring
        if triple_count % 2 == 1:
            in_docstring = not in_docstring
        # Skip if the whole line is inside a multiline string.
        if line_started_in_docstring and triple_count == 0:
            continue
        # Single-line triple-quoted docstring: count=2, stays closed —
        # still skip because the whole body is in the string.
        if triple_count >= 2 and (
            stripped.startswith(('"""', "'''"))
            or stripped.startswith(('r"""', "r'''", 'f"""', "f'''"))
        ):
            continue

        # Comment-only lines: migration notes often reference the old
        # patterns we flag; don't nag on them.
        if stripped.startswith("#"):
            continue

        for rule in rules:
            if _match_rule(rule, line):
                yield LintFinding(
                    path=path,
                    line=line_no,
                    rule=rule,
                    snippet=line.rstrip(),
                )


def check_paths(
    paths: Iterable[str | os.PathLike[str]],
    *,
    target_version: str | None = None,
    min_severity: str = "warning",
) -> list[LintFinding]:
    """Scan ``paths`` and return every finding at or above ``min_severity``.

    Args:
        paths: Files or directories to scan.  Directories recurse.
        target_version: If set, only rules applicable to this version
            are used.  See :func:`rules_for_target` for selection.
        min_severity: ``"warning"`` (default) or ``"error"``.  With
            ``"error"``, warning-level findings are dropped from the
            returned list.

    Returns:
        Findings in deterministic order — files sorted, lines ascending,
        rule IDs ascending within a line.
    """

    rules = rules_for_target(target_version)
    if min_severity == "error":
        rules = tuple(r for r in rules if r.severity == "error")

    findings: list[LintFinding] = []
    # Sort to keep output deterministic across filesystems.
    for file_path in sorted(_iter_python_files(Path(p) for p in paths)):
        findings.extend(_scan_file(file_path, rules))

    findings.sort(key=lambda f: (str(f.path), f.line, f.rule.id))
    return findings
