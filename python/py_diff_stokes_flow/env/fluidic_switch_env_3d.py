import numpy as np

from py_diff_stokes_flow.env.env_base import EnvBase
from py_diff_stokes_flow.common.common import ndarray
from py_diff_stokes_flow.core.py_diff_stokes_flow_core import ShapeComposition3d, StdIntArray3d

class FluidicSwitchEnv3d(EnvBase):
    def __init__(self, seed, folder):
        np.random.seed(seed)

        cell_nums = (32, 32, 16)
        E = 100
        nu = 0.499
        vol_tol = 1e-2
        edge_sample_num = 2
        EnvBase.__init__(self, cell_nums, E, nu, vol_tol, edge_sample_num, folder)

        # Initialize the parametric shapes.
        self._parametric_shape_info = [
            ('polar_bezier3', 27),
            ('bezier', 11),
            ('bezier', 11),
            ('bezier', 11),
        ]
        upper = ndarray([
            [0.0, 0.7],
            [0.1, 0.7],
            [0.75, 0.9],
            [1.0, 0.9],
        ])
        lower = ndarray([
            [1.0, 0.1],
            [0.75, 0.1],
            [0.1, 0.3],
            [0.0, 0.3],
        ])
        right = ndarray([
            [1.0, 0.65],
            [0.9, 0.6],
            [0.9, 0.4],
            [1.0, 0.35],
        ])
        dir_offset = 0.01
        cx, cy, cz = cell_nums
        cxy = ndarray([cx, cy])
        boundary_bezier = ShapeComposition3d()
        boundary_bezier.AddParametricShape('bezier', 11)
        boundary_bezier.AddParametricShape('bezier', 11)
        boundary_bezier.AddParametricShape('bezier', 11)
        params = np.concatenate([
            ndarray(upper * cxy).ravel(),
            ndarray([0, dir_offset, 1]),
            ndarray(lower * cxy).ravel(),
            ndarray([0, -dir_offset, 1]),
            ndarray(right * cxy).ravel(),
            ndarray([dir_offset, 0, 1]),
        ])
        cxyz = StdIntArray3d((int(cx), int(cy), int(cz)))
        boundary_bezier.Initialize(cxyz, params, True)

        # Initialize the node conditions.
        self._node_boundary_info = []
        nx, ny, nz = cx + 1, cy + 1, cz + 1
        inlet_velocity = 5
        for i in range(nx):
            for j in range(ny):
                for k in [0, cz]:
                    self._node_boundary_info.append(((i, j, k, 2), 0))
        for j in range(ny):
            for k in range(nz):
                if boundary_bezier.signed_distance((0, j, k)) < 0:
                    self._node_boundary_info.append(((0, j, k, 0), inlet_velocity))

        # Initialize the interface.
        self._interface_boundary_type = 'no-slip'

        # Initialize the target velocity field.
        on_target_field = np.zeros((ny, nz, 3))
        off_target_field = np.zeros((ny, nz, 3))
        z_half = int(cz // 2)
        y_half = int(cy // 2)
        for j in range(ny):
            for k in range(nz):
                # Skip solid nodes.
                if boundary_bezier.signed_distance((nx - 1, j, k)) >= 0: continue
                if j < y_half:
                    # Outlet 1.
                    if k <= z_half:
                        on_target_field[j, k] = ndarray([inlet_velocity, 0, 0])
                        off_target_field[j, k] = ndarray([0, 0, 0])
                    else:
                        on_target_field[j, k] = ndarray([0, 0, 0])
                        off_target_field[j, k] = ndarray([inlet_velocity, 0, 0])
                else:
                    # Outlet 2.
                    if k <= z_half:
                        on_target_field[j, k] = ndarray([0, 0, 0])
                        off_target_field[j, k] = ndarray([inlet_velocity, 0, 0])
                    else:
                        on_target_field[j, k] = ndarray([inlet_velocity, 0, 0])
                        off_target_field[j, k] = ndarray([0, 0, 0])

        # Other data members.
        self._inlet_velocity = inlet_velocity
        self._boundary_bezier = boundary_bezier
        self._upper = upper
        self._lower = lower
        self._right = right
        self._dir_offset = dir_offset
        self._on_target_field = on_target_field
        self._off_target_field = off_target_field

    def _variables_to_shape_params(self, x):
        x = ndarray(x).copy().ravel()
        assert x.size == 26

        cx, cy, _ = self._cell_nums
        assert cx == cy
        # Since there are two modes, we have two sets of parameters.
        def get_params_and_grads(rotation_angle_idx):
            params = np.concatenate([
                x[:24] * cx,
                ndarray([0.6 * cx, 0.5 * cy, x[rotation_angle_idx]]),
                (self._upper * cx).ravel(),
                ndarray([0.0, self._dir_offset, 1.0]),
                (self._lower * cx).ravel(),
                ndarray([0.0, -self._dir_offset, 1.0]),
                (self._right * cx).ravel(),
                ndarray([self._dir_offset, 0.0, 1.0]),
            ])
            grads = np.zeros((params.size, x.size))
            for i in range(24):
                grads[i, i] = cx
            grads[26, rotation_angle_idx] = 1
            grads = ndarray(grads)
            return params, grads
        params_on, grads_on = get_params_and_grads(24)
        params_off, grads_off = get_params_and_grads(25)
        return [(params_on, grads_on), (params_off, grads_off)]

    def _loss_and_grad_on_velocity_field(self, u):
        assert isinstance(u, list) and len(u) == 2
        nx, ny, nz = self.node_nums()
        assert nx == ny
        def loss_and_grad(u_single, target_field):
            u_field = self.reshape_velocity_field(u_single)
            loss = 0
            grad = np.zeros(u_field.shape)
            cnt = 0
            for j in range(ny):
                for k in range(nz):
                    if self._boundary_bezier.signed_distance((int(nx - 1), int(j), int(k))) < 0:
                        uxyz = u_field[nx - 1, j, k]
                        u_diff = uxyz - target_field[j, k]
                        loss += u_diff.dot(u_diff)
                        grad[nx - 1, j, k] += 2 * u_diff
                        cnt += 1
            loss /= cnt
            grad /= cnt
            return loss, ndarray(grad).ravel()

        loss_and_grad = [
            loss_and_grad(u[0], self._on_target_field),
            loss_and_grad(u[1], self._off_target_field),
        ]

        return loss_and_grad

    def _color_velocity(self, u):
        return float(np.linalg.norm(u) / (self._inlet_velocity * 2))

    def sample(self):
        return np.random.uniform(low=self.lower_bound(), high=self.upper_bound())

    def lower_bound(self):
        return np.concatenate([
            np.full(24, 0.08),
            ndarray([0, -np.pi / 4 * 3])
        ])

    def upper_bound(self):
        return np.concatenate([
            np.full(24, 0.32),
            ndarray([0, -np.pi / 4])
        ])