from timeit import default_timer as timer

from params import params

from cued.hamiltonian.models import BiTe
from cued.main import sbe_solver

def model():
    A = 0.1974      # Fermi velocity

    dirac_system = BiTe(C0=0, C2=0, A=A, R=0, mz=0)
    dirac_system.make_eigensystem()

    return dirac_system

if __name__ == "__main__":
    start = timer()
    sbe_solver(model(), params)
    end = timer()
    print("Time taken: ", end - start, " seconds")