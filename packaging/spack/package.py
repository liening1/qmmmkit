"""Spack package recipe for qmmmkit.

Drop this directory into a custom Spack repository::

    spack repo create ~/.spack/qmmmkit
    cp -r packaging/spack ~/.spack/qmmmkit/packages/qmmmkit
    spack repo add ~/.spack/qmmmkit
    spack install qmmmkit

Once on PyPI, the `pypi` line below is enough; until then `git` is the
authoritative source.
"""

from spack.package import *  # noqa: F401,F403


class Qmmmkit(PythonPackage):
    """QM/MM toolkit: PySCF + OpenMM coupled via ASH, with PES/opt/TS/IRC tasks,
    density-based analyses, and remote SLURM dispatch."""

    homepage = "https://github.com/qmmmkit/qmmmkit"
    pypi = "qmmmkit/qmmmkit-0.1.0.tar.gz"
    git = "https://github.com/qmmmkit/qmmmkit.git"

    license("MIT")

    version("main", branch="main")
    version("0.1.0", sha256="0" * 64)  # replace with the real sha256 after first release

    variant("gui", default=False, description="Install PySide6 desktop GUI")
    variant("slurm", default=True, description="Install SLURM/SSH dispatch dependencies")
    variant("mm", default=True, description="Install OpenMM and forcefield helpers")

    depends_on("python@3.10:", type=("build", "run"))
    depends_on("py-hatchling", type="build")
    depends_on("py-numpy", type=("build", "run"))
    depends_on("py-scipy", type=("build", "run"))
    depends_on("py-matplotlib", type=("build", "run"))
    depends_on("py-pyyaml@6:", type=("build", "run"))
    depends_on("py-rich@13:", type=("build", "run"))
    depends_on("py-typer@0.12:", type=("build", "run"))
    depends_on("py-pyscf@2.5:", type=("build", "run"))
    depends_on("py-mdtraj", type=("build", "run"))
    depends_on("py-parmed", type=("build", "run"))
    depends_on("py-ase", type=("build", "run"))
    depends_on("py-geometric", type=("build", "run"))

    # Optional features
    depends_on("py-openmm@8.1:", when="+mm", type=("build", "run"))
    depends_on("py-paramiko@3:", when="+slurm", type=("build", "run"))
    depends_on("py-pyside6@6.6:", when="+gui", type=("build", "run"))
