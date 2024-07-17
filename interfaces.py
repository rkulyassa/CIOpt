import os
import re
import shutil
from abc import ABC, abstractmethod
import numpy as np

class InterfaceIO(ABC):

    def __init__(self, state_i: str, state_j: str, multiplicity: str, qm_input_file: str, init_geom_file: str):
        self.state_i = state_i
        self.state_j = state_j
        self.multiplicity = multiplicity
        self.qm_input_file = qm_input_file
        self.init_geom_file = init_geom_file

        # Molecular info stored internally
        g = InterfaceIO.parse_geometry(init_geom_file)
        self.num_atoms = g[0]
        self.ground_state_energy = g[1]
        self.atomic_symbols = g[2]
        self.initial_geometry = g[3]

        # Remove existing scratch directory
        if os.path.exists('scr'):
            shutil.rmtree('scr')
        
        # Initialize new scratch directory
        self.init_scr()
        
    @classmethod
    @abstractmethod
    def init_scr(self) -> None:
        ''' Initializes the scratch directory and necessary files. '''
        pass
    
    @staticmethod
    def parse_geometry(file: str) -> list[int, float, list[str], np.ndarray[float]]:
        '''
        Gets geometry-relevant data from an input geometry .xyz file.
        
        Returns:
            list:
                - (int): Number of atoms.
                - (float): Ground state energy.
                - (list[str]): Atomic symbols.
                - (np.ndarray[float]): Initial geometry nuclear coordinates.
        '''

        with open(file, 'r') as f:
            lines = f.readlines()
        
        num_atoms = re.sub(r'\D', '', lines[0])
        ground_state_energy = lines[1][:-1]

        atomic_symbols = []
        geometry = []
        
        for line in lines[2:]:
            data = line.split()
            if not data: continue
            atomic_symbol = data[0]
            atomic_symbols.append(atomic_symbol)
            coordinates = [float(re.sub(r'\n', '', c)) for c in data[1:]]
            geometry.append(coordinates)

        initial_geometry = np.array(geometry, dtype=object)
        
        return [num_atoms, ground_state_energy, atomic_symbols, initial_geometry]

    @classmethod
    @abstractmethod
    def parse_energy(self) -> list[float, float, np.ndarray[float], np.ndarray[float]]:
        '''
        Gets energy-relevant data.
        
        Returns:
            list:
                - (float): Total energy for state I.
                - (float): Total energy for state J.
                - (np.ndarray[float]): Energy gradients for state I.
                - (np.ndarray[float]): Energy gradients for state J.
        '''
        pass

    @staticmethod
    def write_geometry(num_atoms: int, ground_state_energy: float, atomic_symbols: list[str], geometry: np.ndarray[float], output_file: str) -> None:
        ''' Writes the relevant geometry to a .xyz file. '''

        with open(output_file, 'w') as file:
            lines = [num_atoms, ground_state_energy]
            for i, atom in enumerate(atomic_symbols):
                coordinates = ' '.join([f'{c:.8f}' for c in geometry[i]])
                lines.append(f'{atom} {coordinates}')
            file.write('\n'.join(lines))

    @classmethod
    @abstractmethod
    def run_qm(self) -> None:
        ''' Runs a single iteration of the QM program. '''
        pass

class TeraChemIO(InterfaceIO):

    def init_scr(self):
        os.makedirs('scr')
        os.makedirs('scr/GRAD_I')
        os.makedirs('scr/GRAD_J')
        shutil.copy(self.qm_input_file, 'scr/GRAD_I/start.sp')
        shutil.copy(self.qm_input_file, 'scr/GRAD_J/start.sp')
        self.update_qm_input()
        shutil.copy(self.init_geom_file, f'scr/GRAD_I/geom.xyz')
        shutil.copy(self.init_geom_file, f'scr/GRAD_J/geom.xyz')
    
    def parse_geometry(self):
        ''' Wrapper method; reads the current geometry, only reads from GRAD_I since they should be the same for both states. '''
        return super().parse_geometry('scr/GRAD_I/geom.xyz')

    def update_qm_input(self) -> None:
        ''' Updates start.sp file to specify target state and multiplicity. '''

        for state in ['I', 'J']:
            file = f'scr/GRAD_{state}/start.sp'

            with open(file, 'r') as f:
                lines = f.readlines()

            castarget = self.state_i if state == 'I' else self.state_j
            
            for i, line in enumerate(lines):
                if line == '\n': continue
                key = line.strip().split()[0]
                if key == 'castarget':
                    lines[i] = f'castarget               {castarget}               # target state for calculating gradient\n'
                if key == 'castargetmult':
                    lines[i] = f'castargetmult           {self.multiplicity}               # Spin multiplicity\n'
        
            with open(file, 'w') as f:
                f.writelines(lines)
    
    def parse_energy(self, iteration: int):
        ''' Read energy data of states I and J separately. Requires iteration arg since we're keeping scr.geom folders. '''

        scr_index_str = f'.{iteration}' if iteration > 0 else ''

        # Read state I
        with open(f'scr/GRAD_I/scr.geom{scr_index_str}/grad.xyz', 'r') as f:
            lines = f.readlines()

        total_energy_i = float(re.search(r'energy (-?\d+\.\d+)', lines[1]).group(1))
        
        energy_gradients_i = []

        for line in lines[2:]:
            data = line.split()
            coordinates = [float(c) for c in data[1:]]
            if not coordinates: continue
            energy_gradients_i.append(coordinates)
        
        # Read state J
        with open(f'scr/GRAD_J/scr.geom{scr_index_str}/grad.xyz', 'r') as f:
            lines = f.readlines()

        total_energy_j = float(re.search(r'energy (-?\d+\.\d+)', lines[1]).group(1))
        
        energy_gradients_j = []

        for line in lines[2:]:
            data = line.split()
            coordinates = [float(c) for c in data[1:]]
            if not coordinates: continue
            energy_gradients_j.append(coordinates)
        
        return [total_energy_i, total_energy_j, energy_gradients_i, energy_gradients_j]
    
    def write_geometry(self, geometry: np.ndarray[float]) -> None:
        ''' Wrapper method; write geometry to both states' directories. '''

        for state in ['I', 'J']:
            super().write_geometry(self.num_atoms, self.ground_state_energy, self.atomic_symbols, geometry, f'scr/GRAD_{state}/geom.xyz')
    
    def run_qm(self):
        ''' Calls the TeraChem binary for each state. '''

        os.system('cd scr/GRAD_I && terachem start.sp > tera.out')
        os.system('cd scr/GRAD_J && terachem start.sp > tera.out')
    
    # @classmethod
    # def generate_log(self, iterations_count: int, log_file: str) -> None:
    #     '''
    #     Generates log file containing relevant data from each iteration.
    #     For TeraChem, this is possible as a scr.geom folder is created each iteration. Other QM programs may be different.
    #     '''

    #     with open(log_file, 'w') as f:
    #         lines = []
    #         for i in range(iterations_count):
    #             e = self.parse_energy(self, i)
    #             lines.append(f'{i} {e[0]} {e[1]} {e[1] - e[0]}')
    #         f.write('\n'.join(lines))