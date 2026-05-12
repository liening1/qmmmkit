"""Plugin registry tests."""

from __future__ import annotations

import pytest

from qmmmkit._registry import ANALYSES, SCHEDULERS, TASKS, Registry, register_task


def test_builtin_tasks_registered():
    # importing tasks subpackage triggers @register_task decorators
    import qmmmkit.tasks  # noqa: F401

    names = TASKS.names()
    for expected in ("single_point", "optimize", "transition_state", "pes_scan", "irc"):
        assert expected in names


def test_builtin_analyses_registered():
    import qmmmkit.analysis  # noqa: F401

    names = ANALYSES.names()
    for expected in ("population", "natural_orbitals"):
        assert expected in names


def test_builtin_schedulers_registered():
    import qmmmkit.scheduler  # noqa: F401

    names = SCHEDULERS.names()
    assert "local" in names
    # slurm registers when its module is imported (it's in __init__)
    assert "slurm" in names


def test_unknown_name_raises_with_known_list():
    reg: Registry = Registry("widget")
    with pytest.raises(KeyError) as exc:
        reg.get("missing")
    assert "Known: []" in str(exc.value)


def test_decorator_double_register_is_rejected():
    reg: Registry = Registry("widget")
    @reg.register("foo")
    class A: pass
    class B: pass
    with pytest.raises(ValueError):
        reg.register("foo")(B)


def test_decorator_idempotent_for_same_class():
    reg: Registry = Registry("widget")
    @reg.register("foo")
    class A: pass
    # Re-registering the same class to the same name is a no-op (no error).
    reg.register("foo")(A)
