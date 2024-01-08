from params import params

from cued.hamiltonian.models import BiTe
from cued.main import sbe_solver

def model():
    A  = 0.1974      # Fermi velocity
    mz = 0.01837     # prefactor of sigma_z in Hamiltonian

    dirac_system = BiTe(C0=0, C2=0, A=A, R=0, mz=mz)
    dirac_system.make_eigensystem()

    return dirac_system

if __name__ == "__main__":

    sbe_solver(model(), params)