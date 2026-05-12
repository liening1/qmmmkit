"""Plugin registries for tasks, analyses, and schedulers.

Built on Python entry points so external packages can extend qmmmkit
without forking. Registration also works in-process via the
``@register_*`` decorators for the built-in plugins.

Example - adding a new task in an external package::

    # in mypkg/tasks.py
    from qmmmkit.tasks._base import BaseTask
    from qmmmkit._registry import register_task

    @register_task("my_special_method")
    class MySpecialTask(BaseTask):
        spec_class = MySpecialSpec
        def run(self, ...): ...

    # in pyproject.toml
    [project.entry-points."qmmmkit.tasks"]
    my_special_method = "mypkg.tasks:MySpecialTask"
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> class registry. Discovers via entry points + decorators."""

    def __init__(self, kind: str, entry_point_group: str | None = None) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._items: dict[str, type[T]] = {}
        self._loaded_entry_points = False

    # ------------------------------------------------------------
    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            existing = self._items.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"{self.kind} '{name}' already registered to {existing.__module__}.{existing.__name__}"
                )
            self._items[name] = cls
            setattr(cls, "_registry_name", name)
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        self._ensure_entry_points()
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Known: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        self._ensure_entry_points()
        return sorted(self._items)

    def has(self, name: str) -> bool:
        self._ensure_entry_points()
        return name in self._items

    def items(self) -> Iterable[tuple[str, type[T]]]:
        self._ensure_entry_points()
        return list(self._items.items())

    # ------------------------------------------------------------
    def _ensure_entry_points(self) -> None:
        if self._loaded_entry_points or not self.entry_point_group:
            return
        self._loaded_entry_points = True
        try:
            eps = entry_points(group=self.entry_point_group)
        except TypeError:
            # Older importlib.metadata: returns a dict-like
            eps = entry_points().get(self.entry_point_group, [])
        for ep in eps:
            try:
                cls = ep.load()
            except ImportError:
                continue
            self._items.setdefault(ep.name, cls)


# Singletons used throughout the package. They're populated via decorators
# the first time the relevant subpackage (`qmmmkit.tasks`, `qmmmkit.analysis`,
# `qmmmkit.scheduler`) is imported.
TASKS: Registry = Registry("task", "qmmmkit.tasks")
ANALYSES: Registry = Registry("analysis", "qmmmkit.analyses")
SCHEDULERS: Registry = Registry("scheduler", "qmmmkit.schedulers")


def register_task(name: str):
    return TASKS.register(name)


def register_analysis(name: str):
    return ANALYSES.register(name)


def register_scheduler(name: str):
    return SCHEDULERS.register(name)


__all__ = [
    "Registry",
    "TASKS",
    "ANALYSES",
    "SCHEDULERS",
    "register_task",
    "register_analysis",
    "register_scheduler",
]
