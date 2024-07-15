import shutil
import numpy as np
from interfaces import TeraChemIO

DEFAULT_CONSTANTS = {
    'ALPHA': 0.02,          # Hartree
    'SIGMA': 3.5,           # Hartree
    'GAMMA': 0.01,          # Step size
    'STEP_TOL': 0.000001,   # Hartree
    'GRAD_TOL': 0.0005      # Hartree/Bohr
}

def get_objective_gradients(e_i: float, e_j: float, d_e_i: np.ndarray, d_e_j: np.ndarray, alpha: float = DEFAULT_CONSTANTS['ALPHA'], sigma: float = DEFAULT_CONSTANTS['SIGMA']) -> np.ndarray:
    '''
    Based on Levine pg. 407 eq. 7.
    
    Args:
        e_i (float): Total energy of state I.
        e_j (float): Total energy of state J.
        d_e_i (np.ndarray): Matrix (N,3) of energy gradients for each atom in state I.
        d_e_j (np.ndarray): Matrix (N,3) of energy gradients for each atom in state J.
    
    Returns:
        np.ndarray: Matrix (N,3) of objective gradients for each atom.
    '''

    # Merge energy gradient matrices into one matrix (N,2,3). N atoms, 2 states, 3 dimensions
    d_e = np.stack((d_e_i, d_e_j), axis=1)

    # Energy difference
    e_diff = e_i - e_j

    # Energy derivative differences
    d_e_diff = d_e[:, 0, :] - d_e[:, 1, :]

    # Energy derivative averages
    d_e_avg = np.mean(d_e, axis=1)
    
    # Penalty fn. gradient matrix
    d_pen = ((e_diff * e_diff + 2 * alpha * e_diff) / ((e_diff + alpha) * (e_diff + alpha))) * d_e_diff

    # Objective fn. gradient matrix
    d_obj = d_e_avg + sigma * d_pen

    return d_obj

def steepest_gradient_descent(geometry: np.ndarray, gradient: np.ndarray, gamma: float = DEFAULT_CONSTANTS['GAMMA']) -> np.ndarray:
    '''
    Steps a geometric system based on steepest descent gradient method.

    Args:
        geometry (np.ndarray): Matrix (N,3) of nuclear coordinates.
        gradient (np.ndarray): Matrix (N,3) of gradients.
        gamma (float): Step size.

    Returns:
        np.ndarray: Resultant geometry; matrix (N,3) of nuclear coordinates.
    '''

    return geometry + gamma * gradient

if __name__ == '__main__':

    # Read input.in
    input = {}
    with open('./input.in', 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split()
            if data:
                input[data[0]] = data[1]

    # Select and initialize QM interface
    if input['qm_program'] == 'terachem':
        interface = TeraChemIO(input['state_i'], input['state_j'], input['multiplicity'], input['qm_input'], input['init_geom'])

    # Initial QM calculation
    interface.run_qm()
    
    # TODO: convergence criteria
    converged = False
    i = 1

    while not converged and i <= int(input['max_iter']):

        # Get energy data
        e = interface.parse_energy(i)
        e_total_i = e[0]
        e_total_j = e[1]
        e_grad_i = e[2]
        e_grad_j = e[3]

        # Get geometry data
        g = interface.parse_geometry()
        num_atoms = g[0]
        ground_state_energy = g[1]
        atom_symbols = g[2]
        initial_geometry = g[3]

        # Append to log file
        with open(input['log'], 'a') as file:
            file.write(f'{i} {e_total_i} {e_total_j} {e_total_j - e_total_i}\n')

        # Step geometry
        d_obj = get_objective_gradients(e_total_i, e_total_j, e_grad_i, e_grad_j)
        stepped_geometry = steepest_gradient_descent(initial_geometry, d_obj)
        
        # Write geometry
        interface.write_geometry(stepped_geometry, 'scr/GRAD_I/geom.xyz')
        interface.write_geometry(stepped_geometry, 'scr/GRAD_J/geom.xyz')

        # Run QM
        interface.run_qm()
        
        i += 1

    if not input['keep_scr'].lower() == 'yes':
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