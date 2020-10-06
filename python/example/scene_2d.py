import sys
sys.path.append('../')

import numpy as np
from pathlib import Path

from py_diff_stokes_flow.env.random_env_2d import RandomEnv2d
from py_diff_stokes_flow.common.common import ndarray, print_error
from py_diff_stokes_flow.common.grad_check import check_gradients

def test_scene_2d(verbose):
    seed = 42
    env = RandomEnv2d(seed)

    def loss_and_grad(x):
        loss, grad, _ = env.solve(x, { 'solver': 'eigen' })
        return loss, grad

    x0 = env.sample()
    if not check_gradients(loss_and_grad, x0, eps=1e-5, verbose=verbose):
        if verbose:
            print_error('Gradient check in scene_2d failed.')
        return False

    return True

if __name__ == '__main__':
    verbose = True
    test_scene_2d(verbose)