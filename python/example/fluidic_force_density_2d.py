import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.env.fluidic_force_density_env_2d import FluidicForceDensityEnv2d
from py_diff_stokes_flow.common.common import ndarray, print_error
from py_diff_stokes_flow.common.grad_check import check_gradients

def main():
    seed = 42
    solver = 'eigen'
    folder = Path('fluidic_force_density')

    env = FluidicForceDensityEnv2d(seed, folder)
    x = env.sample()
    env.solve(x, False, { 'solver': solver })
    env.render(x, 'vis.png', { 'solver': solver })

if __name__ == '__main__':
    main()