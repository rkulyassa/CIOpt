import numpy as np
import re

# User-defined constants:
ALPHA = 0.02 # Hartree
SIGMA = 3.5 # Hartree
GAMMA = 0.01 # Step size
STEP_TOL = 0.000001 # Hartree
GRAD_TOL = 0.0005 # Hartree/Bohr

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

def parse_energy_gradients(system_output_file: str) -> np.ndarray:
    ''' *Deprecated* Gets energy gradients from TeraChem output '''

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

def get_objective(e_i: float, e_j: float, alpha: float = ALPHA, sigma: float = SIGMA) -> float:
    ''' Based on Levine pg. 407 eq. 3-6. '''

    d_e = e_i - e_j
    pen = (d_e * d_e) / (d_e + alpha)
    e_avg = (e_i + e_j) / 2
    obj = e_avg + sigma * pen
    return obj

def get_objective_gradients(e_i: float, e_j: float, d_e_i: np.ndarray, d_e_j: np.ndarray, alpha: float = ALPHA, sigma: float = SIGMA) -> np.ndarray:
    """
    Based on Levine pg. 407 eq. 7.

    Args:
        e_i (float): The total energy of state I.
        e_j (float): The total energy of state J.
        d_e_i (np.ndarray): The matrix (N,3) of energy gradients for each atom in state I.
        d_e_j (np.ndarray): The matrix (N,3) of energy gradients for each atom in state J.
    
    Returns:
        np.ndarray: The matrix (N,3) of objective gradients for each atom.
    """

    # Merge energy gradient matrices into one matrix (N,2,3). N atoms, 2 states, 3 dimensions
    d_e = np.stack((d_e_i, d_e_j), axis=1)

    # Calculate energy difference
    e_diff = e_i - e_j

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

def write_final_geometry(num_atoms: str, ground_state_energy: str, atoms: list[str], geometry: np.ndarray, output_file: str) -> None:
    with open(output_file, 'w') as file:
        lines = [num_atoms, ground_state_energy]
        for i, atom in enumerate(atoms):
            coordinates = ' '.join([f'{c:.8f}' for c in geometry[i]])
            lines.append(f'{atom} {coordinates}')
        file.write('\n'.join(lines))

if __name__ == '__main__':
    output_file_state_i = './data/ryan_opt_data/s0/tera.o.31613262'
    output_file_state_j = './data/ryan_opt_data/s1/tera.o.31613283'
    geom_file = './data/ryan_opt_data/s0/geom.xyz'
    grad_file_state_i = './data/ryan_opt_data/s0/scr.geom/grad.xyz'
    grad_file_state_j = './data/ryan_opt_data/s1/scr.geom/grad.xyz'


    e_i = parse_energy_data(grad_file_state_i)
    e_j = parse_energy_data(grad_file_state_j)

    d_obj = get_objective_gradients(e_i[0], e_j[0], e_i[1], e_i[1])

    geometry_data = parse_geometry_data(geom_file)
    num_atoms = geometry_data[0]
    ground_state_energy = geometry_data[1]
    atoms = geometry_data[2]
    initial_geometry = geometry_data[3]
    
    final_geometry = steepest_descent(initial_geometry, d_obj)

    write_final_geometry(num_atoms, ground_state_energy, atoms, final_geometry, 'test_out.xyz')
    






    # # Convergence criteria 1 - step tolerance
    # obj = get_objective(e_i, e_j)
    # # store last objective in ./temp.txt, parse from there and compare here
    # prev_obj = 0
    # c1 = np.abs(obj - prev_obj) > STEP_TOL

    # Convergence criteria 2 - parallel gradient component tolerance
    # get unit vector of penalty, dot with d_obj to get c2
    #c2 = (1 / SIGMA) * np.dot(d_obj, u) <= GRAD_TOL

    # Convergence criteria 3 - perpendicular gradient component tolerance
    # c3 = np.abs(d_obj - (d_obj * u))