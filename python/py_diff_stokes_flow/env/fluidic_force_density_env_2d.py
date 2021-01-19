import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray
from matplotlib import collections as mc

class FluidicForceDensityEnv2d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (64, 48)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8), ( 'bezier', 8 ) ]
        # Initialize the node conditions.
        self._node_boundary_info = []
        inlet_velocity = 10
        inlet_range = ndarray([0.45, 0.55])
        inlet_lb, inlet_ub = inlet_range * cell_nums[0]
        outlet_range = ndarray([0.4, 0.6])
        outlet_lb, outlet_ub = outlet_range * cell_nums[1]
        for i in range(cell_nums[0] + 1):
            if inlet_lb < i < inlet_ub:
                self._node_boundary_info.append(((i, 0, 0), 0))
                self._node_boundary_info.append(((i, 0, 1), inlet_velocity))
        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._inlet_range = inlet_range
        self._outlet_range = outlet_range

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()

        # Convert x to the shape parameters.
        left = ndarray([
            [self._inlet_range[0], 0],
            [self._inlet_range[0], self._outlet_range[0]],
            [self._inlet_range[0] / 2, self._outlet_range[0]],
            [0, self._outlet_range[0]]
        ])
        right = ndarray([
            [1, self._outlet_range[0]],
            [(self._inlet_range[1] + 1) / 2, self._outlet_range[0]],
            [self._inlet_range[1], self._outlet_range[0]],
            [self._inlet_range[1], 0]
        ])
        upper = ndarray([
            [0, self._outlet_range[1]],
            [1 / 3, self._outlet_range[1]],
            [2 / 3, self._outlet_range[1]],
            [1, self._outlet_range[1]]
        ])
        # Slightly perturb them to avoid singular boundaries.
        perturb = 0.01
        def get_perturb():
            return np.random.normal(scale=perturb)
        def get_abs_perturb():
            return np.abs(np.random.normal(scale=perturb))
        left[0, 0] += get_perturb()
        left[0, 1] -= get_abs_perturb()
        left[1, 0] += get_perturb()
        left[1, 1] += get_perturb()
        left[2, 0] += get_perturb()
        left[2, 1] += get_perturb()
        left[3, 0] -= get_abs_perturb()
        left[3, 1] += get_perturb()

        right[0, 0] += get_abs_perturb()
        right[0, 1] += get_perturb()
        right[1, 0] += get_perturb()
        right[1, 1] += get_perturb()
        right[2, 0] += get_perturb()
        right[2, 1] += get_perturb()
        right[3, 0] += get_perturb()
        right[3, 1] -= get_abs_perturb()

        upper[0, 0] -= get_abs_perturb()
        upper[0, 1] += get_perturb()
        upper[1, 0] += get_perturb()
        upper[1, 1] += get_perturb()
        upper[2, 0] += get_perturb()
        upper[2, 1] += get_perturb()
        upper[3, 0] += get_abs_perturb()
        upper[3, 1] += get_perturb()

        cxy = ndarray(self._cell_nums)
        left *= cxy
        right *= cxy
        upper *= cxy
        params = np.concatenate([left.ravel(), right.ravel(), upper.ravel()])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad(self, scene, u):
        param_size = self._variables_to_shape_params(self.lower_bound())[0].size
        grad_param = ndarray(np.zeros(param_size))
        return 0, ndarray(np.zeros(u.shape).ravel()), grad_param

    def _render_customized_2d(self, scene, u_field, ax):
        nx, ny = self._node_nums
        lines = []
        scale = 0.1
        for i in range(nx):
            for j in range(ny):
                if scene.GetSignedDistance((i, j)) > 0: continue
                v_begin = ndarray([i, j])
                v_end = v_begin + scale * ndarray(scene.GetFluidicForceDensity((i, j)))
                lines.append((v_begin, v_end))
        ax.add_collection(mc.LineCollection(lines, colors='tab:red', linestyle='-'))

    def sample(self):
        return ndarray(np.zeros(24))