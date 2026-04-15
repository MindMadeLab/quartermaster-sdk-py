"""The v0.2.0 integrator-facing :class:`Result` type.

Wraps :class:`quartermaster_engine.FlowResult` with a nicer public
surface.  ``FlowResult.node_results`` is UUID-keyed (good for debugging
tooling); the :class:`Result` here puts name-keyed ``captures`` front-
and-centre so callers can write ``result["research"].output_text``
instead of iterating ``for uuid, r in flow_result.node_results.items():
if r.node_type == NodeType.INSTRUCTION_FORM: ...``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quartermaster_engine import FlowResult
    from quartermaster_engine.nodes import NodeResult


@dataclass
class Result:
    """Run result — what :func:`run` returns.

    Attributes:
        text: Final text output of the flow.  Equivalent to
            ``FlowResult.final_output``.  For graphs without an End
            node, this is the output of the last finished node.
        captures: Dict of ``capture_as="name"`` → :class:`NodeResult`
            for every node the graph-builder tagged.  Access via
            ``result.captures["notes"].output_text`` or the shorthand
            ``result["notes"]``.
        success: ``True`` when every node finished without a ``STOP``
            strategy error bubbling up.
        error: Concatenated error messages from any failed node, or
            ``None`` on success.
        duration_seconds: Wall-clock time the flow took to execute.
        raw: The underlying :class:`FlowResult` — escape hatch when you
            need UUID-keyed ``node_results`` or the full ``output_data``
            maps.
    """

    text: str
    captures: dict[str, NodeResult] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    duration_seconds: float = 0.0
    raw: FlowResult | None = None

    def __getitem__(self, name: str) -> NodeResult:
        """``result["notes"]`` → the captured NodeResult.

        Raises ``KeyError`` with a list of known capture names on miss
        so the caller can see immediately which key they meant.
        """
        try:
            return self.captures[name]
        except KeyError:
            available = ", ".join(sorted(self.captures)) or "(none)"
            raise KeyError(
                f"No capture named {name!r}. Available captures: {available}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.captures

    @classmethod
    def from_flow_result(cls, fr: FlowResult) -> Result:
        """Build a :class:`Result` from the underlying :class:`FlowResult`."""
        return cls(
            text=fr.final_output,
            captures=dict(fr.captures),
            success=fr.success,
            error=fr.error,
            duration_seconds=fr.duration_seconds,
            raw=fr,
        )


__all__ = ["Result"]
