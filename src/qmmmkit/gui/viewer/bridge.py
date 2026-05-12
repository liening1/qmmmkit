"""QObject exposed to JavaScript via QWebChannel.

Carries atom-pick events and selection-state changes from the NGL canvas
back into Python land where the rest of the app can react to them.
"""

from __future__ import annotations

import json
from typing import Iterable

from PySide6.QtCore import QObject, Signal, Slot


class ViewerBridge(QObject):
    """Bidirectional channel between the NGL page and the Qt host."""

    # JS -> Python
    atomPickedSignal = Signal(dict)
    viewerReadySignal = Signal(int)
    selectionChangedSignal = Signal(list)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    @Slot(str)
    def atomPicked(self, payload: str) -> None:  # noqa: N802 (JS naming)
        try:
            self.atomPickedSignal.emit(json.loads(payload))
        except json.JSONDecodeError:
            pass

    @Slot(int)
    def viewerReady(self, natoms: int) -> None:  # noqa: N802
        self.viewerReadySignal.emit(int(natoms))

    @Slot(str)
    def selectionChanged(self, payload: str) -> None:  # noqa: N802
        try:
            indices = json.loads(payload)
            self.selectionChangedSignal.emit([int(i) for i in indices])
        except json.JSONDecodeError:
            self.selectionChangedSignal.emit([])
