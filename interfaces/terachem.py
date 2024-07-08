import numpy as np
import re

class TeraChem:
    def __init__(self):
        pass

    def update_start_file(start_file: str, state: str, multiplicity: str) -> None:
        with open(start_file, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if line.strip().startswith('castarget'):
                lines[i] = f'castarget               {state}               # target state for calculating gradient\n'
            if line.strip().startswith('spinmult'):
                lines[i] = f'spinmult                {multiplicity}               # Spin multiplicity\n'
        
        with open(start_file, 'w') as file:
            file.writelines(lines)

    def parse_geometry_data(geometry_file: str) -> list[str, str, list[str], np.ndarray]:
        ''' Gets number of atoms, type of atoms, and system coordinates from geom file '''

        with open(geometry_file, 'r') as file:
            lines = file.readlines()
        
        num_atoms = re.sub(r'\D', '', lines[0])
        ground_state_energy = lines[1][:-1]

        atoms = []
        geometry = []
        
        for line in lines[2:]:
            data = line.split()
            atom = data[0]
            atoms.append(atom)
            coordinates = [float(re.sub(r'\n', '', c)) for c in data[1:]]
            geometry.append(coordinates)
        
        return [num_atoms, ground_state_energy, atoms, np.array(geometry, dtype=object)]
    
    def parse_energy_data(gradient_file: str) -> list[float, np.ndarray]:
        ''' Gets energy data (total energy and gradients) from energy gradient file '''
        with open(gradient_file) as file:
            lines = file.readlines()

        total_energy = float(re.search(r'energy (-?\d+\.\d+)', lines[1]).group(1))
        
        energy_gradients = []

        for line in lines[2:]:
            data = line.split()
            coordinates = [float(c) for c in data[1:]]
            energy_gradients.append(coordinates)
        
        return [total_energy, energy_gradients]
    
    def write_final_geometry(num_atoms: str, ground_state_energy: str, atoms: list[str], geometry: np.ndarray, output_file: str) -> None:
        with open(output_file, 'w') as file:
            lines = [num_atoms, ground_state_energy]
            for i, atom in enumerate(atoms):
                coordinates = ' '.join([f'{c:.8f}' for c in geometry[i]])
                lines.append(f'{atom} {coordinates}')
            file.write('\n'.join(lines))