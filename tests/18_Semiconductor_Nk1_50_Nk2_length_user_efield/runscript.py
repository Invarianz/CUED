from params import params
import numpy as np
from numba import njit

from cued.hamiltonian.models import Semiconductor
from cued.main import sbe_solver
from cued.utility.constants import ConversionFactors as co


def make_gaussian(E0, sigma):
    """
    Creates a jitted version of the electric field for fast use inside a solver
    """
    E0 = E0 * co.MVpcm_to_au
    sigma = sigma * co.fs_to_au
    @njit
    def electric_field(t):
        '''
        Returns the instantaneous driving pulse field
        '''
        # Gaussian pulse
        return E0*np.exp(-t**2/sigma**2)

    return electric_field


def model():
    # Hamiltonian Parameters
    A = 2 * co.eV_to_au

    # Gaps used in the dirac system
    mx = 0.05 * co.eV_to_au
    muz = 0.033

    semich_bite_system = Semiconductor(A=A, mz=muz, mx=mx, a=8.28834, nature=True)
    return semich_bite_system


def run(system):

    E0 = 1e-1                        # MV/cm
    sigma = 20                       # fs
    params.electric_field_function = make_gaussian(E0, sigma)
    sbe_solver(system, params)


if __name__ == "__main__":
    run(model())
