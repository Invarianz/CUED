from params import params

from cued.hamiltonian.models import Graphene
from cued.main import sbe_solver

def model():
    t = 0.08                 #t coefficient
    dft_system = Graphene(a=params.a, t=t)
    dft_system.make_eigensystem()

    return dft_system

if __name__ == "__main__":
    sbe_solver(model(), params)
