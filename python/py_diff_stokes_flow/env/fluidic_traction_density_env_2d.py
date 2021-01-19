import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray
from matplotlib import collections as mc

import matplotlib.pyplot as plt

class FluidicTractionDensityEnv2d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (32, 24)
        E = 100
        nu = 0.499
        vol_tol = 1e-3
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [ ('bezier', 8), ('bezier', 8) ]
        # Initialize the node conditions.
        self._node_boundary_info = []
        inlet_velocity = 1
        inlet_range = ndarray([0.125, 0.875])
        inlet_lb, inlet_ub = inlet_range * cell_nums[1]
        for j in range(cell_nums[1] + 1):
            if inlet_lb < j < inlet_ub:
                self._node_boundary_info.append(((0, j, 0), inlet_velocity))
                self._node_boundary_info.append(((0, j, 1), 0))
        # Initialize the interface.
        self._interface_boundary_type = 'free-slip'

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._outlet_velocity = 3 * inlet_velocity
        self._inlet_range = inlet_range

        # Lame parameters.
        self._E = E
        self._nu = nu
        la = E * nu / (1 + nu) / (1 - 2 * nu)
        mu = E / 2 / (1 + nu)
        self._la = la
        self._mu = mu

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 5

        cx, cy = self._cell_nums
        # Convert x to the shape parameters.
        lower = ndarray([
            [1, x[4]],
            x[2:4],
            x[:2],
            [0, self._inlet_range[0]],
        ])
        lower[:, 0] *= cx
        lower[:, 1] *= cy
        upper = ndarray([
            [0, self._inlet_range[1]],
            [x[0], 1 - x[1]],
            [x[2], 1 - x[3]],
            [1, 1 - x[4]],
        ])
        upper[:, 0] *= cx
        upper[:, 1] *= cy
        params = np.concatenate([lower.ravel(), upper.ravel()])

        # Jacobian.
        J = np.zeros((params.size, x.size))
        J[1, 4] = 1
        J[2, 2] = 1
        J[3, 3] = 1
        J[4, 0] = 1
        J[5, 1] = 1
        J[10, 0] = 1
        J[11, 1] = -1
        J[12, 2] = 1
        J[13, 3] = -1
        J[15, 4] = -1
        # Scale it by cx and cy.
        J[:, 0] *= cx
        J[:, 2] *= cx
        J[:, 1] *= cy
        J[:, 3] *= cy
        J[:, 4] *= cy
        return ndarray(params).copy(), ndarray(J).copy()

    def _loss_and_grad(self, scene, u):
        param_size = self._variables_to_shape_params(self.lower_bound())[0].size
        grad_param = ndarray(np.zeros(param_size))
        return 0, ndarray(np.zeros(u.shape).ravel()), grad_param

    def _render_customized_2d(self, scene, u_field, ax):
        nx, ny = self._node_nums
        cx, cy = self._cell_nums
        lines = []

        sample_num = 20
        scale = 0.1
        for i in range(cx):
            for j in range(cy):
                # Only visualize hybrid cells.
                if not scene.IsMixedCell((i, j)): continue

                n = ndarray(scene.GetNormalInMixedCell((i, j)))
                d = scene.GetOffsetInMixedCell((i, j))
                # n.dot(x) + d >= 0 is the solid area.
                n_len = np.linalg.norm(n)
                n /= n_len
                d /= n_len
                assert n[0] != 0 and n[1] != 0

                # Determine the boundary inside this cell.
                # We have a line: n.dot(x) + d = 0.
                # We have a square: [i, i + 1] x [j, j + 1].
                # Compute p0 and p1.
                # Pick an anchor point first.
                # n[0] * x + n[1] * y + d = 0.
                q = ndarray([0, -d / n[1]])
                d = ndarray([n[1], -n[0]])
                # Line equation now becomes q + t * d.
                # q + t * d = [i, ?]
                tx0 = (0 - q[0]) / d[0]
                tx1 = (1 - q[0]) / d[0]
                if tx0 > tx1:
                    tx0, tx1 = tx1, tx0
                # Now tx0 <= tx1.
                # q + t * d = [?, j]
                ty0 = (0 - q[1]) / d[1]
                ty1 = (1 - q[1]) / d[1]
                if ty0 > ty1:
                    ty0, ty1 = ty1, ty0
                # Now ty0 <= ty1.
                t_min = np.max([tx0, ty0])
                t_max = np.min([tx1, ty1])
                assert t_min <= t_max
                p0 = q + t_min * d
                p1 = q + t_max * d
                dl = np.linalg.norm(p1 - p0)
                assert 0 <= np.min(p0) <= np.max(p0) <= 1
                assert 0 <= np.min(p1) <= np.max(p1) <= 1

                # Collect velocity fields.
                u00 = u_field[i, j]
                u01 = u_field[i, j + 1]
                u10 = u_field[i + 1, j]
                u11 = u_field[i + 1, j + 1]

                def compute_F(p):
                    X, Y = p
                    # x = (1 - X) * (1 - Y) * u00 +
                    #     (1 - X) * Y * u01 +
                    #     X * (1 - Y) * u10 +
                    #     XY * u11.
                    F = np.zeros((2, 2))
                    F[:, 0] = -(1 - Y) * u00 - Y * u01 + (1 - Y) * u10 + Y * u11
                    F[:, 1] = -(1 - X) * u00 + (1 - X) * u01 - X * u10 + X * u11
                    return F

                # Integrate along the edge.
                total_traction = 0
                for k in range(sample_num):
                    t = (k + .5) / sample_num
                    p = (1 - t) * p0 + t * p1
                    F = compute_F(p)
                    sigma = self._mu * (F + F.T) + self._la * np.trace(F) * np.eye(2)
                    # Compute traction.
                    total_traction += sigma @ -n * dl / sample_num

                v_begin = ndarray([i, j]) + (p0 + p1) / 2
                v_end = v_begin + scale * total_traction
                lines.append((v_begin, v_end))

        ax.add_collection(mc.LineCollection(lines, colors='tab:orange', linestyle='-'))

    def sample(self):
        return ndarray([0.37398444, 0.38338892, 0.5783081 , 0.37403294, 0.37916777])

    def lower_bound(self):
        return ndarray([0.16, 0.05, 0.49, 0.05, 0.05])

    def upper_bound(self):
        return ndarray([0.49, 0.49, 0.83, 0.49, 0.49])