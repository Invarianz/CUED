<h1>All development on GitHub for this Fork has stopped. I'm hosting all developments on sourcehut instead. See <a href="https://git.sr.ht/~invarianz/cued"> CUED repo</a></h1>

# <img alt="CUED" src="/cued/branding/logo_inverted.png#gh-dark-mode-only" height="100">
# <img alt="CUED" src="/cued/branding/logo.png#gh-light-mode-only" height="100">

| master | devel |
| ------ | ----- |
| ![Test branch parameter](https://github.com/Invarianz/CUED/actions/workflows/regression.yml/badge.svg?=master) | ![Test branch parameter](https://github.com/Invarianz/CUED/actions/workflows/regression.yml/badge.svg?branch=devel) |
| ![Test branch parameter](https://github.com/Invarianz/CUED/actions/workflows/published.yml/badge.svg?=master) | ![Test branch parameter](https://github.com/Invarianz/CUED/actions/workflows/published.yml/badge.svg?branch=devel) |




Package for computing the density matrix dynamics in solids exposed to ultrafast light pulses implementing the Semiconductor Bloch equations (SBE). Includes computation of  k-dependent bandstructures and dipole moments, computation of currents and emission intensity. 

<h3>How to cite and reference CUED</h3>
When using the CUED software package, please cite the following publication:
<br><br>
J. Wilhelm, P. Grössing, A. Seith, J. Crewse, M. Nitsch, L. Weigl, C. Schmid, and F. Evers, <i>Semiconductor-Bloch Formalism: Derivation and Application to High-Harmonic Generation from Dirac Fermions</i>, <a href="https://doi.org/10.1103/PhysRevB.103.125419">Phys. Rev. B <b>103</b>, 125419 (2021)</a>.


<h3>Getting and running the code on Linux</h3>

To download the current version of the code, run

    git clone https://github.com/Invarianz CUED
    
Change to the directory of the code:

    cd CUED
    
Type ``pwd`` and set the outcome as pythonpath:

    export PYTHONPATH=$PYTHONPATH:"/path/to/CUED"

Mandatory files for running the code are ``params.py`` containing the parameters of the calculation and the runscript ``runscript.py``. You can find exemplary parameter files and runscripts in the directory ``tests`` and ``published_calculations``. Now, you can run a test, for example

    cd tests/01_Dirac_Nk1_2_Nk2_2_velocity/
    python3 runscript.py
    
The code is MPI parallel, you can also run it via

    mpirun -np 2 python3 runscript.py

The output is written to ``time_data.dat`` (time-dependent current) and ``frequency_data.dat`` (emission spectrum). If you set ``save_latex_pdf = True``
 in ``params.py`` and if pdflatex is installed on your Linux machine, CUED will generate ``latex_pdf_files/CUED_summary.pdf`` containing plots of the bandstructure, dipoles, Brillouin zone, current, emission spectrum, ... 
