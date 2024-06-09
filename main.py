from reader import Reader
from writer import Writer
import numpy as np
from numpy.typing import NDArray

# User-defined constants:
ALPHA = 0.02 # Hartree
SIGMA = 3.5 # Hartree
STEP_TOL = 0.000001 # Hartree
GRAD_TOL = 0.0005 # Hartree/Bohr

def step(R: NDArray[np.float64], e_I: float, e_J: float):

    # Calculate average energy of states I and J
    e_avg = (e_I + e_J) / 2

    # Calculate energy difference of states I and J
    e_diff = e_I - e_J

    # Evaluate penalty function
    penalty = (e_diff * e_diff) / (e_diff + ALPHA)

    # Evaluate objective function
    objective = e_avg + SIGMA * penalty

    # TODO: Determine next state with gradient of objective fn.

if __name__ == '__main__':
    grad_file = './GRAD/scr.geom/grad.xyz'
    geom_file = './out.xyz'
    reader = Reader(grad_file, geom_file)
    geometry = reader.parse_geom()
    step(geometry, 5, 6)