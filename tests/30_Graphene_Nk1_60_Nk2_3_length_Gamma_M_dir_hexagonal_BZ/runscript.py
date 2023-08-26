from params import params

from cued.hamiltonian import Graphene
from cued.main import sbe_solver

def model():
    t = 0.08                 #t coefficient
    dft_system = cued.hamiltonian.Graphene(a=params.a, t=t)
    return dft_system

def run(system):
    sbe_solver(system, params)
    return 0

if __name__ == "__main__":
    run(model())
