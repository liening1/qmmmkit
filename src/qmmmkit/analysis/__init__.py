"""Analysis plugins. Importing this package registers the built-ins."""

from ._base import BaseAnalysis
from .natural_orbitals import (
    NaturalOrbitalsAnalysis,
    natural_orbitals,
    write_molden,
)
from .population import (
    PopulationAnalysis,
    population_analysis,
)
from .charge_displacement import (
    charge_displacement,
)
from .density_difference import (
    DensityDifferenceAnalysis,
    density_difference_cube,
)
from .nci import NCIAnalysis
from .orbital_cube import OrbitalCubeAnalysis

__all__ = [
    "BaseAnalysis",
    "NaturalOrbitalsAnalysis",
    "PopulationAnalysis",
    "DensityDifferenceAnalysis",
    "NCIAnalysis",
    "OrbitalCubeAnalysis",
    "natural_orbitals",
    "population_analysis",
    "charge_displacement",
    "density_difference_cube",
    "write_molden",
]
