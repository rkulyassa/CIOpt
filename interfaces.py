import numpy as np
import re

class TeraChemIO:
    def __init__(self):
        pass

    def update_start_file(start_file: str, state: str, multiplicity: str) -> None:
        with open(start_file, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if line == '\n': continue
            key = line.strip().split()[0]
            if key == 'castarget':
                lines[i] = f'castarget               {state}               # target state for calculating gradient\n'
            if key == 'castargetmult':
                lines[i] = f'castargetmult           {multiplicity}               # Spin multiplicity\n'
        
        with open(start_file, 'w') as file:
            file.writelines(lines)

    # def parse_geometry(file: str) -> list[list[str], np.ndarray[float], list[float]]:
    #     ''' Gets type of atoms, system coordinates, and masses from .geometry file '''

    #     with open(file, 'r') as f:
    #         lines = f.readlines()
        
    #     atoms = []
    #     geometry = []
    #     masses = []

    #     for line in lines[5:]:
    #         data = line.split()
    #         if not data: continue
    #         atom = data[0]
    #         atoms.append(atom)
    #         coordinates = [float(c) for c in data[1:3]]
    #         geometry.append(coordinates)
    #         mass = float(data[4])
    #         masses.append(mass)
        
    #     return [atoms, np.array(geometry, dtype=object), masses]

    def parse_geometry(file: str) -> list[str, str, list[str], np.ndarray[float]]:
        ''' Gets number of atoms, type of atoms, and system coordinates from .xyz file '''

        with open(file, 'r') as f:
            lines = f.readlines()
        
        num_atoms = re.sub(r'\D', '', lines[0])
        ground_state_energy = lines[1][:-1]

        atoms = []
        geometry = []
        
        for line in lines[2:]:
            data = line.split()
            if not data: continue
            atom = data[0]
            atoms.append(atom)
            coordinates = [float(re.sub(r'\n', '', c)) for c in data[1:]]
            geometry.append(coordinates)
        
        return [num_atoms, ground_state_energy, atoms, np.array(geometry, dtype=object)]
    
    def parse_energy_gradient(file: str) -> list[float, np.ndarray[float]]:
        ''' Gets energy data (total energy and gradients) from energy gradient file '''
        with open(file) as f:
            lines = f.readlines()

        total_energy = float(re.search(r'energy (-?\d+\.\d+)', lines[1]).group(1))
        
        energy_gradients = []

        for line in lines[2:]:
            data = line.split()
            coordinates = [float(c) for c in data[1:]]
            if not coordinates: continue
            energy_gradients.append(coordinates)
        
        return [total_energy, energy_gradients]
    
    def write_geometry(num_atoms: str, ground_state_energy: str, atoms: list[str], geometry: np.ndarray[float], output_file: str) -> None:
        with open(output_file, 'w') as file:
            lines = [num_atoms, ground_state_energy]
            for i, atom in enumerate(atoms):
                coordinates = ' '.join([f'{c:.8f}' for c in geometry[i]])
                lines.append(f'{atom} {coordinates}')
            file.write('\n'.join(lines))