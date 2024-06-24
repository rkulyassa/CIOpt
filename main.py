import numpy as np
import re

# User-defined constants:
ALPHA = 0.02 # Hartree
SIGMA = 3.5 # Hartree
STEP_TOL = 0.000001 # Hartree
GRAD_TOL = 0.0005 # Hartree/Bohr

def parse_system_gradient(gradient_file: str) -> np.ndarray:
    with open(gradient_file, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(re.sub(r'\D', '', lines[0])) # necessary?

    gradient = []
    
    for line in lines[2:]:
        data = line.split()
        atom = data[0] # necessary?
        coordinates = [float(c) for c in data[1:]]
        gradient.append(coordinates)

    return np.array(gradient, dtype=object)

def parse_geometry(geometry_file: str) -> np.ndarray:
    with open(geometry_file, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(re.sub(r'\D', '', lines[0])) # necessary?

    geometry = []
    
    for line in lines[2:]:
        data = line.split()
        atom = data[0] # necessary?
        coordinates = [float(re.sub(r'\n', '', c)) for c in data[1:]]
        geometry.append(coordinates)
    
    return np.array(geometry, dtype=object)

def parse_energy_states(system_output_file: str) -> np.ndarray:
    with open(system_output_file, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while not lines[i].__contains__('Total Energy (a.u.)'):
        i += 1
    
    f = i
    while not lines[f] == '\n':
        f += 1

    energy_states = []

    for line in lines[i+2:f]:
        data = line.split()
        energy = float(data[2])
        energy_states.append(energy)
    
    return np.array(energy_states, dtype=object)


def parse_energy_derivatives(system_output_file: str) -> np.ndarray:
    with open(system_output_file, 'r') as file:
        lines = file.readlines()

    i = 0
    while not lines[i].startswith('Gradient units'):
        i += 1
    
    f = i
    while not lines[f].startswith('Net gradient'):
        f += 1
    
    energy_derivatives = []

    for line in lines[i+3:f-1]:
        data = line.split()
        coordinates = [float(c) for c in data]
        energy_derivatives.append(coordinates)
    
    return np.array(energy_derivatives, dtype=object)

def get_objective_gradient(e_I: float, e_J: float, d_e: np.ndarray) -> np.ndarray:
    """
    Based on Levine pg. 407 eq. 7

    Args:
        e_I (float): The total energy of state I
        e_J (float): The total energy of state J
        d_e (np.ndarray): The matrix (shape: Nx2x3) of energy derivatives. N atoms, 2 states, 3 dimensions
    
    Returns:
        np.ndarray: The objective gradient for each atom (shape: Nx3).
    """

    # Calculate energy difference
    e_diff = e_I - e_J

    # Calculate energy derivative differences
    d_e_diff = d_e[:, 0, :] - d_e[:, 1, :]

    # Calculate energy derivative averages
    d_e_avg = np.mean(d_e, axis=1)
    
    # Evaluate penalty fn.
    d_pen = ((e_diff * e_diff + 2 * ALPHA * e_diff) / ((e_diff + ALPHA) * (e_diff + ALPHA))) * d_e_diff

    # Evaluate objective fn.
    d_obj = d_e_avg + SIGMA * d_pen

    # print('Energy difference:', e_diff)
    # print('Energy derivative differences:', d_e_diff)
    # print('Energy derivative averages:', d_e_avg)
    # print('Penalty gradient:', d_pen)
    # print('Objective gradient:', d_obj)

    return d_obj

def steepest_descent(geometry: np.ndarray, gradient: np.ndarray):
    """
    Steps a geometric system based on steepest descent gradient method.

    Args:

    """



if __name__ == '__main__':
    energy_states = parse_energy_states('./GRAD/geom.out')
    energy_derivatives = parse_energy_derivatives('./GRAD/geom.out')
    state_i = energy_states[0]
    state_j = energy_states[1]
    objective_gradient = get_objective_gradient(state_i, state_j, energy_derivatives)

    print(objective_gradient)

    # system_gradient = parse_system_gradient('./GRAD/scr.geom/grad.xyz')
    # print(system_gradient)
    # objective_gradient = get_objective_gradient(5, 8, np.array([[1,2,2], [3,1,5]]))
    # print(objective_gradient)
    # geometry = parse_geometry('./GRAD/geom.xyz')
    # print(geometry)