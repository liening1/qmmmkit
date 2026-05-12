"""Application entry point.

Spawns a QApplication, applies the dark stylesheet, and shows the main window.
Use ``python -m qmmmkit.gui`` or the ``qmmmkit-gui`` console script.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    return run(argv)


def run(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    QCoreApplication.setOrganizationName("qmmmkit")
    QCoreApplication.setApplicationName("qmmmkit")

    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle("Fusion")
    app.setFont(_default_font())

    qss_path = Path(__file__).parent / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    from .main_window import MainWindow

    window = MainWindow()
    window.resize(1500, 950)
    window.show()
    return app.exec()


def _default_font():
    from PySide6.QtGui import QFont, QFontDatabase

    candidates = ["Inter", "Segoe UI Variable", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial"]
    families = set(QFontDatabase.families())
    family = next((c for c in candidates if c in families), "Sans Serif")
    f = QFont(family, 10)
    f.setHintingPreference(QFont.PreferFullHinting)
    return f


if __name__ == "__main__":
    sys.exit(run())
