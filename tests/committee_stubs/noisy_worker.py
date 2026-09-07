"""A worker whose model load prints to stdout, like MACE/SevenNet/CHGNet do.

Not a test module: it is launched as a subprocess by
tests/test_committee_worker.py. The print happens inside
``_build_calculator``, i.e. *after* ``_protect_stdout`` has run, which is
exactly where a real library banner appears.
"""
import sys

from ase.calculators.emt import EMT

from mliprun.core.committee import worker


def _noisy_build(request):
    print("BANNER: loading model weights")
    sys.stdout.write("more chatter from the library\n")
    sys.stdout.flush()
    return EMT()


worker._build_calculator = _noisy_build

if __name__ == "__main__":
    sys.exit(worker.main())
