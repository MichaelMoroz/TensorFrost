from typing import List, Tuple, NamedTuple
import math
import numpy as np

class Vector3:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other: 'Vector3') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __mul__(self, other: float) -> 'Vector3':
        return Vector3(self.x * other, self.y * other, self.z * other)

class Vector4:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __mul__(self, other: float) -> 'Vector4':
        return Vector4(self.x * other, self.y * other, self.z * other, self.w * other)

class Orbital(NamedTuple):
    atom_id: int
    qnum_n: int
    qnum_l: int
    qnum_m: int

class Atom:
    ELEMENT_NAMES = {
        1: "Hydrogen", 2: "Helium", 3: "Lithium", 4: "Beryllium", 5: "Boron",
        6: "Carbon", 7: "Nitrogen", 8: "Oxygen", 9: "Fluorine", 10: "Neon",
        11: "Sodium", 12: "Magnesium", 13: "Aluminum", 14: "Silicon", 15: "Phosphorus",
        16: "Sulfur", 17: "Chlorine", 18: "Argon", 19: "Potassium", 20: "Calcium",
        21: "Scandium", 22: "Titanium", 23: "Vanadium", 24: "Chromium", 25: "Manganese",
        26: "Iron", 27: "Cobalt", 28: "Nickel", 29: "Copper", 30: "Zinc"
    }

    AtomCount = [0] * (len(ELEMENT_NAMES) + 1)

    def __init__(self, element_type: int, name: str, position: Vector3):
        self.element_type = element_type
        self.name = name
        self.position = position

    def electron_configuration(self):
        electron_configuration = ""
        for n, l, m, e in self.get_electron_configuration():
            electron_configuration += f"n={n} l={l} m={m} e={e}\n"
        print(electron_configuration)

    def get_charge(self) -> float:
        return float(self.element_type)

    @staticmethod
    def add_electron(configuration: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if configuration:
            last_orbital = configuration[-1]
            if last_orbital[3] < 2 and last_orbital[2] == last_orbital[1]:
                orbital_index = next((i for i, orbital in enumerate(configuration) if orbital[3] == 1), -1)
                if orbital_index != -1:
                    orbital = configuration[orbital_index]
                    configuration[orbital_index] = (orbital[0], orbital[1], orbital[2], orbital[3] + 1)
                else:
                    configuration[-1] = (last_orbital[0], last_orbital[1], last_orbital[2], last_orbital[3] + 1)
            elif last_orbital[2] < last_orbital[1]:
                configuration.append((last_orbital[0], last_orbital[1], last_orbital[2] + 1, 1))
            else:
                orbital_sequence = [
                    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1),
                    (5, 0), (4, 2), (5, 1), (6, 0), (4, 3), (5, 2), (6, 1), (7, 0),
                    (5, 3), (6, 2), (7, 1)
                ]
                orbital_index = orbital_sequence.index((last_orbital[0], last_orbital[1]))
                next_orbital = orbital_sequence[orbital_index + 1]
                configuration.append((next_orbital[0], next_orbital[1], -next_orbital[1], 1))
        else:
            configuration.append((1, 0, 0, 1)) # 1s1
        return configuration

    def get_electron_configuration(self, ionization_level: int = 0) -> List[Tuple[int, int, int, int]]:
        atomic_number = self.element_type
        electronic_structure = []
        for _ in range(atomic_number - ionization_level):
            electronic_structure = self.add_electron(electronic_structure)
        return electronic_structure
    
class Molecule:
    def __init__(self, atoms: List[Atom], name = "", target_energy = 0.0, ionization_level: int = 0, orbitals_per_electron: int = 1):
        self.atoms = atoms
        self.ionization_level = ionization_level
        self.orbitals_per_electron = orbitals_per_electron
        self.orbitals: List[Orbital] = []
        self.orbital_array: List[Vector4] = []
        self.spin_up_electrons = 0
        self.spin_down_electrons = 0
        self.electron_count = 0
        self.target_energy = target_energy
        self.name = name
        self.initialize_orbitals()

    def initialize_orbitals(self):
        self.spin_up_electrons = 0
        self.spin_down_electrons = 0

        for i, atom in enumerate(self.atoms):
            atom_ionization_level = self.ionization_level if i == 0 else 0
            atom_orbitals = atom.get_electron_configuration(atom_ionization_level)
            invert_atom_spin = self.spin_up_electrons > self.spin_down_electrons

            for orbital in atom_orbitals:
                for _ in range(self.orbitals_per_electron):
                    self.orbitals.append(Orbital(atom_id=i, qnum_n=1, qnum_l=0, qnum_m=0))

                if orbital[3] == 1:  # if 1 electron in orbital then it is spin up
                    if invert_atom_spin:
                        self.spin_down_electrons += 1
                    else:
                        self.spin_up_electrons += 1
                elif orbital[3] == 2:  # if 2 electrons in orbital then it is spin up and spin down
                    self.spin_up_electrons += 1
                    self.spin_down_electrons += 1

        orbital_list = [
            (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1), (3, 0, 0),
            (3, 1, 0), (3, 1, 1), (3, 2, 0), (3, 2, 1), (3, 2, 2)
        ]

        self.orbital_array = [Vector4(orbital.atom_id, orbital.qnum_n, orbital.qnum_l, orbital.qnum_m) 
                              for orbital in self.orbitals]

        self.electron_count = self.spin_up_electrons + self.spin_down_electrons

    def get_atoms(self) -> np.ndarray:
        atom_array = [[atom.position.x, atom.position.y, atom.position.z, atom.get_charge()] for atom in self.atoms]
        return np.array(atom_array, dtype=np.float32)
    
    def get_orbitals(self, add_per_atom = 0, multiply_per_atom = 1) -> np.ndarray:
        orbs = [orbital.atom_id for orbital in self.orbitals]

        # multiply per atom orbitals (just repeat the orbitals)
        orbs = orbs * multiply_per_atom

        # add per atom orbitals
        for i, atom in enumerate(self.atoms):
            for _ in range(add_per_atom):
                orbs.append(i)

        return np.array(orbs, dtype=np.int32)
        
    
    def get_summary(self):
        summary = "Molecule: {}\n".format(self.name)
        dict_atoms = {}
        for atom in self.atoms:
            if atom.name in dict_atoms:
                dict_atoms[atom.name] += 1
            else:
                dict_atoms[atom.name] = 1
        
        structure = ""
        for atom_name, atom_count in dict_atoms.items():
            structure += "{}{} ".format(atom_name, atom_count)

        summary += "Structure: {}\n".format(structure)
        summary += "Electron count: {}\n".format(self.electron_count)
        summary += "Spin up electrons: {}\n".format(self.spin_up_electrons)
        summary += "Spin down electrons: {}\n".format(self.spin_down_electrons)
        summary += "Target energy: {} Ha\n".format(self.target_energy)

        return summary
        
        
