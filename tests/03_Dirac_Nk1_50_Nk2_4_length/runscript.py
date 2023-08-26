from params import params

from cued.hamiltonian.models
from cued.main import sbe_solver

def model():
	A = 0.1974      # Fermi velocity

	dirac_system = BiTe(C0=0, C2=0, A=A, R=0, mz=0)

	return dirac_system

def run(system):

	sbe_solver(system, params)

	return 0

if __name__ == "__main__":
	run(model())
