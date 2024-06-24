import numpy as np
import re

# User-defined constants:
ALPHA = 0.02 # Hartree
SIGMA = 3.5 # Hartree
GAMMA = 0.01 # Step size
STEP_TOL = 0.000001 # Hartree
GRAD_TOL = 0.0005 # Hartree/Bohr

def parse_system_gradients(gradient_file: str) -> np.ndarray:
    """
    Args:
        gradient_file (str): The gradient file to read from.
    
    Returns:
        np.ndarray: The matrix (N,3) of gradients for each atom.
    """

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
    """
    Args:
        geometry_file (str): The geometry file to read from.
    
    Returns:
        np.ndarray: The matrix (N,3) of nuclear coordinates.
    """

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

def parse_total_energies(system_output_file: str) -> list[float]:
    """
    Args:
        system_output_file (str): The system output file to read from.
    
    Returns:
        list[float]: The list of total energies for each energy state.
    """

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
    
    return energy_states

def parse_energy_gradients(system_output_file: str) -> np.ndarray:
    """
    Args:
        system_output_file (str): The system output file to read from.
    
    Returns:
        np.ndarray: The matrix (N,3) of energy gradients for each atom.
    """
    with open(system_output_file, 'r') as file:
        lines = file.readlines()

    i = 0
    while not lines[i].startswith('Gradient units'):
        i += 1
    
    f = i
    while not lines[f].startswith('Net gradient'):
        f += 1
    
    energy_gradients = []

    for line in lines[i+3:f-1]:
        data = line.split()
        coordinates = [float(c) for c in data]
        energy_gradients.append(coordinates)
    
    return np.array(energy_gradients, dtype=object)

def get_objective_gradients(e_I: float, e_J: float, d_e_I: np.ndarray, d_e_J: np.ndarray, alpha: float = ALPHA, sigma: float = SIGMA) -> np.ndarray:

    """
    Based on Levine pg. 407 eq. 7.

    Args:
        e_I (float): The total energy of state I.
        e_J (float): The total energy of state J.
        d_e_I (np.ndarray): The matrix (N,3) of energy gradients for each atom in state I.
        d_e_J (np.ndarray): The matrix (N,3) of energy gradients for each atom in state J.
    
    Returns:
        np.ndarray: The matrix (N,3) of objective gradients for each atom.
    """

    # Merge energy gradient matrices into one matrix (N,2,3). N atoms, 2 states, 3 dimensions
    d_e = np.stack((d_e_I, d_e_J), axis=1)

    # Calculate energy difference
    e_diff = e_I - e_J

    # Calculate energy derivative differences
    d_e_diff = d_e[:, 0, :] - d_e[:, 1, :]

    # Calculate energy derivative averages
    d_e_avg = np.mean(d_e, axis=1)
    
    # Evaluate penalty fn.
    d_pen = ((e_diff * e_diff + 2 * alpha * e_diff) / ((e_diff + alpha) * (e_diff + alpha))) * d_e_diff

    # Evaluate objective fn.
    d_obj = d_e_avg + sigma * d_pen

    # print('Energy difference:', e_diff)
    # print('Energy derivative differences:', d_e_diff)
    # print('Energy derivative averages:', d_e_avg)
    # print('Penalty gradient:', d_pen)
    # print('Objective gradient:', d_obj)

    return d_obj

def steepest_descent(geometry: np.ndarray, gradient: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """
    Steps a geometric system based on steepest descent gradient method.

    Args:
        geometry (np.ndarray): The matrix (N,3) of nuclear coordinates.
        gradient (np.ndarray): The matrix (N,3) of energy gradients.
        gamma (float): The step size.

    Returns:
        np.ndarray: The resultant geometry; matrix (N,3) of nuclear coordinates.
    """

    return geometry - gamma * gradient

if __name__ == '__main__':
    energies = parse_total_energies('./GRAD/geom.out')
    e_i = energies[0]
    e_j = energies[1]

    d_e_i = parse_energy_gradients('./GRAD/geom.out')
    d_e_j = parse_energy_gradients('./GRAD/geom.out') # need energy gradients for 2nd state
    objective_gradient = get_objective_gradients(e_i, e_j, d_e_i, d_e_j)

    print(objective_gradient)

    initial_geometry = parse_geometry('./GRAD/geom.xyz')
    final_geometry = steepest_descent(initial_geometry, objective_gradient)