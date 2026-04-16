"""v0.4.0 regression: ``qm.configure(telemetry=True)`` is sugar for
``qm.configure(...); qm.telemetry.instrument()``.

Round-2 wishlist ("Telemetry as a configure kwarg") —
two boot-time calls collapse to one.
"""

from __future__ import annotations

from unittest.mock import patch

import quartermaster_sdk as qm


def test_configure_telemetry_false_does_not_instrument() -> None:
    """The default (False) preserves v0.3.x behaviour — no auto-instrument."""
    with patch.object(qm.telemetry, "instrument") as instrument_spy:
        qm.configure(provider="ollama", default_model="gemma4:26b")

    instrument_spy.assert_not_called()


def test_configure_telemetry_true_calls_instrument() -> None:
    """``telemetry=True`` invokes ``qm.telemetry.instrument()`` once."""
    with patch.object(qm.telemetry, "instrument") as instrument_spy:
        qm.configure(provider="ollama", default_model="gemma4:26b", telemetry=True)

    instrument_spy.assert_called_once_with()


def test_configure_telemetry_kwarg_is_keyword_only() -> None:
    """``telemetry`` cannot be passed positionally — keyword-only by design.

    Guards against accidental positional misuse like
    ``qm.configure("ollama", "http://...", "key", "model", None, True)``
    silently triggering instrumentation.
    """
    import inspect

    sig = inspect.signature(qm.configure)
    telemetry_param = sig.parameters["telemetry"]
    assert telemetry_param.kind == inspect.Parameter.KEYWORD_ONLY
    assert telemetry_param.default is False
