"""CLI entry point for ``python -m quartermaster_sdk.lint``.

Subcommands:
    check [--target-version V] [--severity S] PATH...
    list-rules
    show-rule QMxxx
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .checker import check_paths
from .rules import all_rules, get_rule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quartermaster_sdk.lint",
        description=(
            "Quartermaster SDK semantic-change linter. Catches "
            "API-churn patterns pre-commit."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Scan files/directories for rule matches.")
    p_check.add_argument(
        "--target-version",
        default=None,
        help=(
            "SDK version being targeted (e.g. '0.4.0'). Limits rules "
            "to ones that apply at that version."
        ),
    )
    p_check.add_argument(
        "--severity",
        choices=("warning", "error"),
        default="warning",
        help=(
            "Minimum severity to report. 'warning' (default) reports "
            "everything; 'error' reports only errors."
        ),
    )
    p_check.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to scan. Directories recurse.",
    )

    sub.add_parser("list-rules", help="Print every curated rule (id + summary).")

    p_show = sub.add_parser(
        "show-rule", help="Print a single rule's full migration advice."
    )
    p_show.add_argument("rule_id", help="Rule ID, e.g. QM001.")

    return parser


def _cmd_check(args: argparse.Namespace) -> int:
    findings = check_paths(
        args.paths,
        target_version=args.target_version,
        min_severity=args.severity,
    )
    if not findings:
        return 0
    for f in findings:
        print(f.format())
    # Exit 1 as soon as anything fires at the requested severity —
    # pre-commit picks this up as a hook failure.
    return 1


def _cmd_list_rules(_args: argparse.Namespace) -> int:
    rules = all_rules()
    # Plain-text table: id, severity, summary.  Stable for grep.
    for rule in rules:
        status = ""
        if rule.reverted_in is not None:
            status = f"  (reverted in {rule.reverted_in})"
        print(f"{rule.id} [{rule.severity}] {rule.summary}{status}")
    return 0


def _cmd_show_rule(args: argparse.Namespace) -> int:
    try:
        rule = get_rule(args.rule_id)
    except KeyError as exc:
        # exc.args[0] already carries the full "Unknown rule id ..."
        # message with the list of valid IDs.
        print(str(exc.args[0]), file=sys.stderr)
        return 2
    print(rule.advice)
    return 0


_DISPATCH = {
    "check": _cmd_check,
    "list-rules": _cmd_list_rules,
    "show-rule": _cmd_show_rule,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Programmatic entry: returns an exit code, does not call ``sys.exit``."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits with 2 on bad args; convert to a return value
        # so tests can assert on it without catching SystemExit.
        code = exc.code if isinstance(exc.code, int) else 2
        return code
    handler = _DISPATCH[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
