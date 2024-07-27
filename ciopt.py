import shutil
import numpy as np
from interfaces import TeraChemIO

DEFAULT_CONSTANTS = {
    'ALPHA': 0.02,          # Hartree
    'SIGMA': 3.5,           # Hartree
    'DELTA': 0.001,         # Hartee, initial energy gap required for CI
    'GAMMA': 0.01,          # Step size
    'STEP_TOL': 0.000001,   # Hartree
    'GRAD_TOL': 0.0005      # Hartree/Bohr
}

def levine_method(e_i: float, e_j: float, d_e_i: np.ndarray[float], d_e_j: np.ndarray[float], alpha: float = DEFAULT_CONSTANTS['ALPHA'], sigma: float = DEFAULT_CONSTANTS['SIGMA']) -> list[float, float, np.ndarray[float], np.ndarray[float]]:
    '''
    Based on Levine pg. 407 eq. 7.
    
    Args:
        e_i (float): Total energy of state I.
        e_j (float): Total energy of state J.
        d_e_i (np.ndarray[float]): Matrix (N,3) of energy gradients for each atom in state I.
        d_e_j (np.ndarray[float]): Matrix (N,3) of energy gradients for each atom in state J.
    
    Returns:
        float: The value of the objective fn.
        float: The value of the penalty fn.
        np.ndarray[float]: Matrix (N,3) of objective gradients for each atom.
        np.ndarray[float]: Matrix (N,3) of penalty gradients for each atom.
    '''

    # Energy difference
    e_diff = e_i - e_j

    # Energy derivative differences
    # d_e_diff = d_e[:, 0, :] - d_e[:, 1, :]
    d_e_diff = d_e_i - d_e_j

    # Energy derivative averages
    # d_e_avg = np.mean(d_e, axis=1)
    d_e_avg = (d_e_i + d_e_j) / 2.0

    # Penalty fn. gradient matrix
    d_pen = ((e_diff ** 2 + 2 * alpha * e_diff) / ((e_diff + alpha) ** 2)) * d_e_diff

    # Objective fn. gradient matrix
    d_obj = d_e_avg + sigma * d_pen

    # Evaluate penalty & objective fns.
    pen = (e_diff * e_diff) / (e_diff + alpha)
    obj = (e_i + e_j)/2 + sigma * pen

    return [obj, pen, d_obj, d_pen]


def steepest_gradient_descent(geometry: np.ndarray[float], gradient: np.ndarray[float], gamma: float = DEFAULT_CONSTANTS['GAMMA']) -> np.ndarray[float]:
    '''
    Steps a geometric system based on steepest descent gradient method.

    Args:
        geometry (np.ndarray[float]): Matrix (N,3) of nuclear coordinates.
        gradient (np.ndarray[float]): Matrix (N,3) of gradients.
        gamma (float): Step size.

    Returns:
        np.ndarray[float]: Resultant geometry; matrix (N,3) of nuclear coordinates.
    '''

    return geometry + gamma * gradient

def check_convergence(prior_obj: np.ndarray[float], obj: np.ndarray[float], d_obj: np.ndarray[float], d_pen: np.ndarray[float], sigma: float = DEFAULT_CONSTANTS['SIGMA'], step_tol: float = DEFAULT_CONSTANTS['STEP_TOL'], grad_tol: float = DEFAULT_CONSTANTS['GRAD_TOL']) -> bool:
    c1 = c2 = c3 = False

    # Change in objective
    c1 = np.abs(prior_obj - obj) <= step_tol

    # Unit vector along penalty
    u = d_pen / np.linalg.norm(d_pen, axis=1, keepdims=True)

    # Component of objective parallel to penalty direction
    d_obj_parallel = (1 / sigma) * np.sum(d_obj * u, axis=1)
    c2 = np.all(np.abs(d_obj_parallel) <= grad_tol)

    # Component of objective perpendicular to penalty direction
    d_obj_perpendicular = d_obj - np.sum(d_obj * u, axis=1, keepdims=True) * u
    c3 = np.any(np.linalg.norm(d_obj_perpendicular, axis=1) <= grad_tol)

    print(c1, c2, c3)
    return c1 and c2 and c3

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
    
    # Runtime vars
    converged = False
    prior_obj = None # keeps track of the objective in the previous iteration, used in convergence criteria
    i = 0

    while not converged and i < int(input['max_iter']):

        # Get energy data
        e = interface.parse_energy(i)
        e_total_i = e[0]
        e_total_j = e[1]
        e_grad_i = e[2]
        e_grad_j = e[3]

        with open(input['log'], 'a') as f:
            f.write(f'{i} {e_total_i} {e_total_j} {e_total_j - e_total_i}\n')

        # Get current geometry
        current_geometry = interface.parse_geometry()[3]

        # Calculate objective and penalty gradients
        levine_data = levine_method(e_total_i, e_total_j, e_grad_i, e_grad_j, float(input['alpha']), float(input['sigma']))
        obj = levine_data[0]
        pen = levine_data[1]
        d_obj = levine_data[2]
        d_pen = levine_data[3]

        if e_total_j - e_total_i >= DEFAULT_CONSTANTS['DELTA']:
            DEFAULT_CONSTANTS['SIGMA'] += 1
            print(DEFAULT_CONSTANTS['SIGMA'])

        # Check convergence criteria
        if i > 0:
            if check_convergence(prior_obj, obj, d_obj, d_pen, float(input['step_tol']), float(input['sigma'])):
                converged = True
                print(f'Converged after {i} iterations.')
                break

        prior_obj = obj
        stepped_geometry = steepest_gradient_descent(current_geometry, d_obj)
        
        # Write geometry
        interface.write_geometry(stepped_geometry)

        # Run QM
        interface.run_qm()
        
        i += 1

    if input['keep_scr'].lower() == 'no':
        shutil.rmtree('scr')