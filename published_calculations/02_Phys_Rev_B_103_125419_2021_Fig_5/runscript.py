from params import params

from cued.hamiltonian.models import TwoSiteSemiconductor
from cued.main import sbe_solver

import numpy as np

def model():

    a = 6            # lattice constant in atomic units
    t = 3.0/27.211   # hopping: 3 eV
    m = 3.0/27.211   # on-site energy difference of two sites: 3 eV

    params.length_BZ_E_dir = 2*np.pi/a

    semiconductor_system = TwoSiteSemiconductor(lattice_const=a, hopping=t, onsite_energy_difference=m)
    semiconductor_system.make_eigensystem()

    return semiconductor_system

if __name__ == "__main__":
    sbe_solver(model(), params)
