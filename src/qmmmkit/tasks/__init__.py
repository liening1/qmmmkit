"""Task plugins. Importing this package registers the built-in tasks."""

from ._base import BaseTask
from .singlepoint import SinglePointTask
from .optimize import OptimizeTask, TransitionStateTask
from .pes_scan import PESScanTask
from .irc import IRCTask
from .neb import NEBTask
from .eda import EDATask

__all__ = [
    "BaseTask",
    "SinglePointTask",
    "OptimizeTask",
    "TransitionStateTask",
    "PESScanTask",
    "IRCTask",
    "NEBTask",
    "EDATask",
]
