from utils import *
from atom import *

angstrom_to_bohr = 1.88973

atomH = Atom(1, "H", Vector3(3.015*0.5, 0.0, 0.0))
atomLi = Atom(3, "Li", Vector3(-3.015*0.5, 0.0, 0.0))
lih_molecule = Molecule([atomH, atomLi], "LiH", target_energy = -8.070548)

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

atomO = Atom(8, "O", Vector3(0.0, 0.0, 0.1173)*angstrom_to_bohr)
atomH1 = Atom(1, "H", Vector3(0.0, -0.7572, -0.4696)*angstrom_to_bohr)
atomH2 = Atom(1, "H", Vector3(0.0, 0.7572, -0.4696)*angstrom_to_bohr)
h2o_molecule = Molecule([atomO, atomH1, atomH2], "Water", target_energy = -75.0104)

atomZn = Atom(30, "Zn", Vector3(0.0, 0.0, 0.0))
zn_molecule = Molecule([atomZn], "Zinc", target_energy = -257.026)

atomN = Atom(7, "N", Vector3(0.0, 0.0, 0.22013))
atomH1 = Atom(1, "H", Vector3(0.0, 1.77583, -0.51364))
atomH2 = Atom(1, "H", Vector3(1.53791, -0.88791, -0.51364))
atomH3 = Atom(1, "H", Vector3(-1.53791, -0.88791, -0.51364))
nh3_molecule = Molecule([atomN, atomH1, atomH2, atomH3], "Ammonia", target_energy = -56.56295)

atomC = Atom(6, "C", Vector3(0.0517, 0.7044, 0.0))
atomN = Atom(7, "N", Vector3(0.0517, -0.7596, 0.0))
atomH1 = Atom(1, "H", Vector3(-0.9417, 1.1762, 0.0))
atomH2 = Atom(1, "H", Vector3(-0.4582, -1.0994, 0.8124))
atomH3 = Atom(1, "H", Vector3(-0.4582, -1.0994, -0.8124))
atomH4 = Atom(1, "H", Vector3(0.5928, 1.0567, 0.8807))
atomH5 = Atom(1, "H", Vector3(0.5928, 1.0567, -0.8807))
ch3nh2_molecule = Molecule([atomC, atomN, atomH1, atomH2, atomH3, atomH4, atomH5], "Methylamine", target_energy = -95.8554)

atomC1 = Atom(6, "C", Vector3(2.2075, -0.7566, 0.0))
atomC2 = Atom(6, "C", Vector3(0.0, 1.0572, 0.0))
atomO = Atom(8, "O", Vector3(-2.2489, -0.4302, 0.0))
atomH1 = Atom(1, "H", Vector3(-3.6786, 0.7210, 0.0))
atomH2 = Atom(1, "H", Vector3(0.0804, 2.2819, 1.6761))
atomH3 = Atom(1, "H", Vector3(0.0804, 2.2819, -1.6761))
atomH4 = Atom(1, "H", Vector3(3.9985, 0.2736, 0.0))
atomH5 = Atom(1, "H", Vector3(2.1327, -1.9601, 1.6741))
atomH6 = Atom(1, "H", Vector3(2.1327, -1.9601, -1.6741)) 
c2h5oh_molecule = Molecule([atomC1, atomC2, atomO, atomH1, atomH2, atomH3, atomH4, atomH5, atomH6], "Ethanol", target_energy = -155.0308)


atomC1 = Atom(6, "C", Vector3(0.0, 2.13792, 0.58661))
atomC2 = Atom(6, "C", Vector3(0.0, -2.13792, 0.58661))
atomC3 = Atom(6, "C", Vector3(1.41342, 0.0, -0.58924))
atomC4 = Atom(6, "C", Vector3(-1.41342, 0.0, -0.58924))
atomH1 = Atom(1, "H", Vector3(0.0, 2.33765, 2.64110))
atomH2 = Atom(1, "H", Vector3(0.0, 3.92566, -0.43023))
atomH3 = Atom(1, "H", Vector3(0.0, -2.33765, 2.64110))
atomH4 = Atom(1, "H", Vector3(0.0, -3.92566, -0.43023))
atomH5 = Atom(1, "H", Vector3(2.67285, 0.0, -2.19514))
atomH6 = Atom(1, "H", Vector3(-2.67285, 0.0, -2.19514))
c4h6_molecule = Molecule([atomC1, atomC2, atomC3, atomC4, atomH1, atomH2, atomH3, atomH4, atomH5, atomH6], "Bicyclobutane", target_energy = -155.9263)
