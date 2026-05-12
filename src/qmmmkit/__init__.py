"""qmmmkit: QM/MM workflows on PySCF + OpenMM via ASH, with density-based analyses."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qmmmkit")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

from .system import QMMMSystem
from .calculator import QMMMCalculator

__all__ = ["QMMMSystem", "QMMMCalculator", "__version__"]
