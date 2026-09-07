"""A worker whose forces are deliberately biased relative to plain EMT.

Not a test module: launched as a subprocess (via ``RemoteMember(argv=...)``)
so one committee member genuinely disagrees with a plain-EMT sibling, for
exercising the disagreement-flagging path end to end. ``noisy_worker.py``
established the override point used here (``worker._build_calculator``); this
stub reuses it to change what the calculator *returns* rather than what it
prints.

The bias (``BIAS_EV_PER_A``) is large relative to a typical EMT force on a
rattled lattice (order 0.1-1 eV/A), so the resulting sigma_max is unambiguous
regardless of the geometry reached.
"""
import sys

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT

from mliprun.core.committee import worker

BIAS_EV_PER_A = 2.0


class _BiasedEMT(Calculator):
    """Plain EMT with a fixed force offset added on every atom's x-component."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._emt = EMT()

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._emt.calculate(atoms=atoms, properties=properties,
                            system_changes=system_changes)
        forces = np.array(self._emt.results["forces"], dtype=float)
        forces[:, 0] += BIAS_EV_PER_A
        energy = self._emt.results["energy"]
        self.results["energy"] = energy
        self.results["free_energy"] = self._emt.results.get(
            "free_energy", energy)
        self.results["forces"] = forces


def _biased_build(request):
    return _BiasedEMT()


worker._build_calculator = _biased_build

if __name__ == "__main__":
    sys.exit(worker.main())
