"""A worker that loads, answers N calc requests, then exits without replying.

Not a test module: launched as a subprocess to exercise the loud abort on a
dead member, and (later, in the optimize integration) partial-results
survival. N comes from MLIPRUN_STUB_DIE_AFTER, default 0.
"""
import os
import sys

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from mliprun.core.committee import protocol, worker


def main():
    out = worker._protect_stdout()
    stdin = sys.stdin.buffer
    remaining = int(os.environ.get("MLIPRUN_STUB_DIE_AFTER", "0"))
    calc = None
    while True:
        line = stdin.readline()
        if not line:
            return 0
        request = protocol.decode(line)
        if request.get("cmd") == "load":
            calc = EMT()
            out.write(protocol.encode(protocol.ok(t_load=0.0, versions={})))
        elif request.get("cmd") == "calc":
            if remaining <= 0:
                # Die mid-conversation, without replying.
                os._exit(7)
            remaining -= 1
            atoms = Atoms(numbers=request["numbers"],
                          positions=request["positions"],
                          cell=request["cell"], pbc=request["pbc"])
            atoms.calc = calc
            out.write(protocol.encode(protocol.ok(
                energy=float(atoms.get_potential_energy()),
                forces=np.asarray(atoms.get_forces()).tolist(),
                t_calc=0.0)))
        else:
            out.write(protocol.encode(protocol.ok()))


if __name__ == "__main__":
    sys.exit(main())
