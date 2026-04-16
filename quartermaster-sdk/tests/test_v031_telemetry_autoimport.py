"""v0.3.1 regression: ``import quartermaster_sdk as qm; qm.telemetry.X``
must work without an explicit ``from quartermaster_sdk import telemetry``.

The v0.3.0 wheel shipped ``telemetry.py`` as a sibling module but
didn't import it inside ``__init__.py``, so ``qm.telemetry`` raised
``AttributeError`` even though the playbook + examples all show the
``qm.telemetry.instrument()`` form. v0.3.1 adds the one-line
``from . import telemetry`` to fix this; the test below is the
regression guard.
"""

from __future__ import annotations


def test_telemetry_attribute_exists_after_top_level_import() -> None:
    """``qm.telemetry`` resolves without any extra import."""
    import quartermaster_sdk as qm

    assert hasattr(qm, "telemetry"), (
        "qm.telemetry attribute missing — `from . import telemetry` "
        "may have been removed from quartermaster_sdk/__init__.py"
    )


def test_telemetry_callables_reachable_via_top_level_alias() -> None:
    """Both `instrument` and `uninstrument` are reachable via the
    ``qm.telemetry.X`` form documented in the README + playbook."""
    import quartermaster_sdk as qm

    assert callable(qm.telemetry.instrument)
    assert callable(qm.telemetry.uninstrument)


def test_telemetry_module_identity_matches_explicit_import() -> None:
    """The ``qm.telemetry`` attribute is the same module object you'd
    get from ``from quartermaster_sdk import telemetry`` — not a stub
    or alias of a different module."""
    import quartermaster_sdk as qm
    from quartermaster_sdk import telemetry as telemetry_explicit

    assert qm.telemetry is telemetry_explicit
