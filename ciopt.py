import os
import shutil
import numpy as np
from interfaces.terachem import TeraChem

# User-defined constants:
ALPHA = 0.02 # Hartree
SIGMA = 3.5 # Hartree
GAMMA = 0.01 # Step size
STEP_TOL = 0.000001 # Hartree
GRAD_TOL = 0.0005 # Hartree/Bohr

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
    # print(d_e_diff)

    # Calculate energy derivative averages
    d_e_avg = np.mean(d_e, axis=1)
    
    # Get penalty fn. gradients
    d_pen = ((e_diff * e_diff + 2 * alpha * e_diff) / ((e_diff + alpha) * (e_diff + alpha))) * d_e_diff

    # Get objective fn. gradients
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

    return geometry + gamma * gradient

if __name__ == '__main__':

    # Read inputs
    input = {}
    with open('./input.in', 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split()
            if data:
                input[data[0]] = data[1]

    interface = input['interface']
    template_file = input['template']
    state_i = input['state_i']
    state_j = input['state_j']
    multiplicity = input['multiplicity']
    geom_file = input['init_geom']
    out_file = input['out_geom']
    log_file = input['log']
    max_iter: int = int(input['max_iter'])
    keep_scr: bool = input['keep_scr'].lower() == 'yes'

    # Create scratch dir & files
    if not os.path.exists('scr'):
        os.makedirs('scr')
        os.makedirs('scr/GRAD1')
        os.makedirs('scr/GRAD2')
    shutil.copy(template_file, 'scr/GRAD1/start.sp')
    shutil.copy(template_file, 'scr/GRAD2/start.sp')
    shutil.copy(geom_file, 'scr/GRAD1/geom.xyz')
    shutil.copy(geom_file, 'scr/GRAD2/geom.xyz')
    TeraChem.update_start_file('scr/GRAD1/start.sp', state_i, multiplicity)
    TeraChem.update_start_file('scr/GRAD2/start.sp', state_j, multiplicity)

    # Initial QM calculation
    os.system('cd scr/GRAD1 && terachem start.sp > tera.out')
    os.system('cd scr/GRAD2 && terachem start.sp > tera.out')
    
    # TODO: convergence criteria
    converged = False
    i = 0

    while not converged and i < max_iter:

        # Parse data
        scr_index = i if i > 0 else ''
        e_i = TeraChem.parse_energy_data(f'scr/GRAD1/scr.geom{scr_index}/grad.xyz')
        e_j = TeraChem.parse_energy_data(f'scr/GRAD2/scr.geom{scr_index}/grad.xyz')
        e_total_i = e_i[0]
        e_total_j = e_j[0]
        e_grad_i = e_i[1]
        e_grad_j = e_j[1]
        geom_data = TeraChem.parse_geometry_data(geom_file)
        num_atoms = geom_data[0]
        ground_state_energy = geom_data[1]
        atoms = geom_data[2]
        initial_geometry = geom_data[3]

        # Step geometry
        d_obj = get_objective_gradients(e_total_i, e_total_j, e_grad_i, e_grad_j)
        final_geometry = steepest_descent(initial_geometry, d_obj)

        TeraChem.write_final_geometry(num_atoms, ground_state_energy, atoms, final_geometry, 'scr/GRAD1/geom.xyz')
        TeraChem.write_final_geometry(num_atoms, ground_state_energy, atoms, final_geometry, 'scr/GRAD2/geom.xyz')

        # Run QM
        os.system('cd scr/GRAD1 && terachem start.sp > tera.out')
        os.system('cd scr/GRAD2 && terachem start.sp > tera.out')

        # Append to log file
        with open(log_file, 'a') as file:
            file.write(f'{i} {e_total_i} {e_total_j} {e_total_j - e_total_i}\n')
        
        i += 1

    if not keep_scr:
        shutil.rmtree('scr')





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