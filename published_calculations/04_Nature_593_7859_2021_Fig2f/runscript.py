from params import params

from cued.hamiltonian.models import Semiconductor
from cued.main import sbe_solver
from cued.utility.constants import eV_to_au

def model():
    # Hamiltonian Parameters
    A = 2 * eV_to_au

    # Gaps used in the dirac system
    mx = 0.05 * eV_to_au
    muz = 0.033

    semich_bite_system = Semiconductor(A=A, mz=muz, mx=mx, a=params.a, nature=True)
    semich_bite_system.make_eigensystem()

    return semich_bite_system

if __name__ == "__main__":
    sbe_solver(model(), params)
