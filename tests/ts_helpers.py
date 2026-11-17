"""A one-atom double-well calculator for TS-verification tests.

The energy acts on atom 0 relative to ``centre``:

    E = a (x^2 - 1)^2 + by y^2 + bz z^2        (eV, Angstrom)

With the defaults the saddle is at ``centre`` (one imaginary mode along x),
the minima are at ``centre`` +- (1, 0, 0), and the barrier is ``a`` = 1 eV.
``by=-2.0`` makes y a second imaginary direction.
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

CENTRE = np.array([5.0, 5.0, 5.0])


class DoubleWell(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, a=1.0, by=2.0, bz=2.0, centre=CENTRE, **kwargs):
        super().__init__(**kwargs)
        self.a, self.by, self.bz = a, by, bz
        self.centre = np.asarray(centre, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y, z = self.atoms.positions[0] - self.centre
        energy = self.a * (x**2 - 1.0) ** 2 + self.by * y**2 + self.bz * z**2
        forces = np.zeros((len(self.atoms), 3))
        forces[0] = [-4.0 * self.a * x * (x**2 - 1.0), -2.0 * self.by * y, -2.0 * self.bz * z]
        self.results = {"energy": energy, "forces": forces}


def single_atom(position):
    """One H atom at ``position`` in a 10 Angstrom cubic cell, no periodicity."""
    return Atoms("H", positions=[position], cell=[10.0, 10.0, 10.0], pbc=False)
