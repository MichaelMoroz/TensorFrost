from ast import literal_eval
from numpy import ndarray
from utils import *
from atom import *

# for parsing .xyz file comment without manually writing a complete parser
from shlex import shlex,split

# contains some useful functions (e.g. rotation, overlay, formulae, renumbering, etc.)
from xyz_py import lab_to_num, load_xyz, load_xyz_comment

def read_xyz(filename, use_custom_metadata=True, verbosity=3):
	"""
	Parameters
	----------
	filename: str
		Filename of the .xyz file to load
	use_custom_metadata: bool
		If True, parameters will be loaded from the second line (comment) of the XYZ file.
		
		Expected Format:
		
		```
		TF parameter1=value1 parameter2=value2 ...
		```
		
		Format Example:
		
		```
		TF name="Bicyclobutane" target_energy=-155.9263
		```
	verbosity: int
		How verbose to be when printing file loading statistics
		<=0: Nothing will be printed
		  1: Files being loaded will be printed
		  2: Files and statistics (e.g. name, # of atoms, etc.) will be printed
		>=3: Every concievable detail (including custom metadata) about the file loading process will be printed (default)
	"""

	if(verbosity > 0):
		print("load_xyz.py: Loading \""+filename+"\".")

	molecule_name = filename

	energy = -1.0

	(labels, coords) = load_xyz(filename)

	num_atoms = len(labels)

	if(verbosity > 1):
		print("\tAtoms: "+str(num_atoms))

	number = lab_to_num(labels)

	coords = [Vector3(x[0], x[1], x[2]) for x in coords]

	atoms = [Atom(number[i], labels[i], coords[i]) for i in range(num_atoms)]

	if(use_custom_metadata):
		if(verbosity > 2):
			print("\tUse Custom Metadata: "+str(use_custom_metadata))

		comment = load_xyz_comment(filename)

		#print(comment[:3])

		if(comment[:3] == 'TF '):
			if(verbosity > 1):
				print("\tMetadata Found: True")

			#print(comment)

			#comment = comment.strip()

			# remove the "magic number"
			comment = comment[3:]

			# Split parameters
			metadata = split(comment)

			#for param in metadata:
			#	print(param+"\n")

			for param in metadata:
				'''
				TODO: Shlex the Parameter
				The comment could contain '=' which will be split, but this will just result in the text after
				the '=' being cut off.
				'''
				#shlex(param)
				key = param.split('=')

				if(len(key) < 2):
					raise Exception("Invalid parameter \'"+param+"\' in "+str(filename)+":\n\t"+comment)

				match key[0]:
					case 'name':
						molecule_name = str(key[1])
					case 'target_energy':
						energy = float(key[1])
					case 'comment':
						if(verbosity > 2):
							print("\tComment:")

							comment_str = literal_eval('\"'+key[1]+'\"')

							lines = comment_str.splitlines()

							for line in lines:
								print("\t\t"+line)

					case _:
						raise Exception("Invalid parameter \'"+param+"\' in "+str(filename)+":\n\t"+comment)
		else:
			# custom metadata not used
			if(verbosity > 2):
				print("\tMetadata Found: False")
	else:
		# Custom metadata not found
		if(verbosity > 2):
			print("\tUse Custom Metadata: "+str(use_custom_metadata))

	if(verbosity > 1):
		print("\tMolecule Name: "+str(molecule_name))

	if energy <= 0.0:
		if(verbosity > 1):
			print("\tTarget Energy: Unspecified")

		# energy was not set, so let the Molecule() class handle it
		return Molecule(atoms, molecule_name)
	
	if(verbosity > 1):
			print("\tTarget Energy: "+str(energy))

	return Molecule(atoms, molecule_name, target_energy=energy)
