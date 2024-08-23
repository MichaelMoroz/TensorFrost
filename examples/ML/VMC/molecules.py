from utils import *
from atom import *

atom1 = Atom(3, "Li", Vector3(-5.051*0.5, 0.0, 0.0))
atom2 = Atom(3, "Li", Vector3(5.051*0.5, 0.0, 0.0))
li2_molecule = Molecule([atom1, atom2], "Li2", target_energy = -14.9966)

atom = Atom(6, "C", Vector3(0.0, 0.0, 0.0))
c_molecule = Molecule([atom], "Carbon", target_energy = -37.846772)

atomC = Atom(6, "C", Vector3(0.0, 0.0, 0.0))
atomH1 = Atom(1, "H", Vector3(1.18886, 1.18886, 1.18886))
atomH2 = Atom(1, "H", Vector3(-1.18886, -1.18886, 1.18886))
atomH3 = Atom(1, "H", Vector3(-1.18886, 1.18886, -1.18886))
atomH4 = Atom(1, "H", Vector3(1.18886, -1.18886, -1.18886))
ch4_molecule = Molecule([atomC, atomH1, atomH2, atomH3, atomH4], "Methane", target_energy = -40.51400)

atom = Atom(8, "O", Vector3(0.0, 0.0, 0.0))
o_molecule = Molecule([atom], "Oxygen", target_energy = -75.06655)

atom = Atom(10, "Ne", Vector3(0.0, 0.0, 0.0))
ne_molecule = Molecule([atom], "Neon", target_energy = -128.9366)

atomC1 = Atom(6, "C", Vector3(0.0, 0.0, 1.26135))
atomC2 = Atom(6, "C", Vector3(0.0, 0.0, -1.26135))
atomH1 = Atom(1, "H", Vector3(0.0, 1.74390, 2.33889))
atomH2 = Atom(1, "H", Vector3(0.0, -1.74390, 2.33889))
atomH3 = Atom(1, "H", Vector3(0.0, 1.74390, -2.33889))
atomH4 = Atom(1, "H", Vector3(0.0, -1.74390, -2.33889))
c2h4_molecule = Molecule([atomC1, atomC2, atomH1, atomH2, atomH3, atomH4], "Ethane", target_energy = -78.5844)