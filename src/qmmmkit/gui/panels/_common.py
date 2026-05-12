"""Common widget helpers used across sidebar panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget


def card(title: str | None = None, parent: QWidget | None = None) -> tuple[QFrame, QVBoxLayout]:
    """Create a styled "card" container, optionally with a section title.

    Returns (card_widget, inner_layout) so the caller can add child widgets
    to ``inner_layout``.
    """
    frame = QFrame(parent)
    frame.setProperty("role", "card")
    frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(8)
    if title:
        label = QLabel(title.upper(), frame)
        label.setProperty("role", "section")
        layout.addWidget(label)
    return frame, layout


def section_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setProperty("role", "section")
    return label


def hint_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setProperty("role", "hint")
    label.setWordWrap(True)
    return label


def title_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setProperty("role", "title")
    return label
