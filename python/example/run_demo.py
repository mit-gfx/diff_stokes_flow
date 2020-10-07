import sys
sys.path.append('../')

import numpy as np
from importlib import import_module
import scipy.optimize
import time
import matplotlib.pyplot as plt

from py_diff_stokes_flow.common.common import print_info, print_ok, print_error, print_warning, ndarray
from py_diff_stokes_flow.common.grad_check import check_gradients

# Update this dictionary if you would like to add new demos.
all_demo_names = {
    # ID: (module name, class name).
    'amplifier': ('amplifier_env_2d', 'AmplifierEnv2d'),
}

if __name__ == '__main__':
    # Input check.
    if len(sys.argv) != 2:
        print_error('Usage: python run_demo.py [demo_name]')
        sys.exit(0)
    demo_name = sys.argv[1]
    assert demo_name in all_demo_names

    # Hyperparameters.
    seed = 42
    sample_num = 16
    solver = 'eigen'
    rel_tol = 1e-4
    max_iter = 10
    enable_grad_check = True

    # Load class.
    module_name, env_name = all_demo_names[demo_name]
    Env = getattr(import_module('py_diff_stokes_flow.env.{}'.format(module_name)), env_name)
    env = Env(seed, demo_name)

    # Global search: randomly sample initial guesses and pick the best.
    losses = []
    best_sample = None
    best_loss = np.inf
    for _ in range(sample_num):
        x = env.sample()
        loss, _ = env.solve(x, False, { 'solver': solver }) 
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_sample = np.copy(x)
    print_info('Randomly sampled {:d} initial guesses.'.format(sample_num))
    print_info('Loss (min, max, mean): ({:4f}, {:4f}, {:4f}).'.format(
        np.min(losses), np.max(losses), np.mean(losses)
    ))
    unit_loss = np.mean(losses)

    # Local optimization: run L-BFGS from best_sample.
    x_init = np.copy(best_sample)
    bounds = scipy.optimize.Bounds(env.lower_bound(), env.upper_bound())
    def loss_and_grad(x):
        t_begin = time.time()
        loss, grad, _ = env.solve(x, True, { 'solver': solver })
        # Normalize loss and grad.
        loss /= unit_loss
        grad /= unit_loss
        t_end = time.time()
        print('loss: {:3.6e}, |grad|: {:3.6e}, time: {:3.6f}s'.format(loss, np.linalg.norm(grad), t_end - t_begin))
        return loss, grad

    if enable_grad_check:
        print_info('Checking gradients...')
        # Sanity check gradients.
        success = check_gradients(loss_and_grad, x_init)
        if success:
            print_ok('Gradient check succeeded.')
        else:
            print_error('Gradient check failed.')
            sys.exit(0)

    loss, grad = loss_and_grad(x_init)
    opt_history = [(x_init.copy(), loss, grad.copy())]
    def callback(x):
        loss, grad = loss_and_grad(x)
        global opt_history
        opt_history.append((x.copy(), loss, grad.copy()))
        print_info('Summary of iteration {:4d}'.format(len(opt_history)))
        print_info('loss: {:3.6e}, |grad|: {:3.6e}, |x|: {:3.6e}'.format(
            loss, np.linalg.norm(grad), np.linalg.norm(x)))

    results = scipy.optimize.minimize(loss_and_grad, x_init.copy(), method='L-BFGS-B', jac=True, bounds=bounds,
        callback=callback, options={ 'ftol': rel_tol, 'maxiter': max_iter})
    if not results.success:
        print_warning('Local optimization fails to reach the optimal condition and will return the last solution.')
    x_final = results.x

    # Plot the optimization progress.
    plt.rc('pdf', fonttype=42)
    plt.rc('font', size=22)
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=22)

    fig = plt.figure(figsize=(18, 10))
    ax_loss = fig.add_subplot(121)
    ax_grad = fig.add_subplot(122)

    ax_loss.set_position((0.07, 0.2, 0.39, 0.6))
    iterations = np.arange(len(opt_history))
    ax_loss.plot(iterations, [l for _, l, _ in opt_history], color='tab:red')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_yscale('log')
    ax_loss.grid(True, which='both')

    ax_grad.set_position((0.54, 0.2, 0.39, 0.6))
    ax_grad.plot(iterations, [np.linalg.norm(g) + np.finfo(np.float).eps for _, _, g in opt_history],
        color='tab:green')
    ax_grad.set_xlabel('Iteration')
    ax_grad.set_ylabel('|Gradient|')
    ax_grad.set_yscale('log')
    ax_grad.grid(True, which='both')

    plt.show()
    fig.savefig('{}/progress.pdf'.format(demo_name))

    # Render the results.
    print_info('Rendering optimization history in {}/'.format(demo_name))
    for k, (xk, _, _) in enumerate(opt_history):
        env.render(xk, '{:04d}.png'.format(k), { 'solver': solver })
        print_info('{}/{:04d}.png is ready.'.format(demo_name, k))