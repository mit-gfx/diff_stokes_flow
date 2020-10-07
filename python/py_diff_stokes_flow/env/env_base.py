import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import collections as mc

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import StdRealVector, Scene2d, Scene3d
from py_diff_stokes_flow.common.common import ndarray, create_folder, print_warning

# For each derived class, it only needs to initialize three data members in __init__ and overlead the loss and grad
# function _loss_and_grad_on_velocity_field. A typical derived may look like this:
#
# class EnvDerived:
#     def __init__(self, ...):
#         EnvBase.__init__(self, ...)
#
#         self._parametric_shape_info = {}
#         self._node_boundary_info = []
#         self._interface_boundary_type = 'free-slip'
#
#     def _variables_to_shape_params(self, var):
#         return var, np.eye(var.size)
#
#     def _loss_and_grad_on_velocity_field(self, u):
#         u_tensor = self.reshape_velocity_field(u)
#         loss = ...
#         grad = ...
#         return loss, grad
#
#     def sample(self):
#         return np.random.uniform(self.lower_bound(), self.upper_bound())
#
#     def lower_bound(self):
#         return ndarray([-np.inf, -np.inf, ...])
#
#     def upper_bound(self):
#         return ndarray([np.inf, np.inf, ...])
#
# The derived class is typically used as follows:
# env = EnvDerived(...)
# loss, grad, info = env.solve(x)
# scene = info['scene']
# u = info['velocity_field']

class EnvBase:
    def __init__(self,
        cell_nums,          # 2D or 3D int array.
        E,                  # Young's modulus.
        nu,                 # Poisson's ratio.
        vol_tol,            # vol_tol <= mixed cell <= 1 - vol_tol.
        edge_sample_num,    # The number of samples inside each cell's axis.
        folder              # The folder that stores temporary files.
    ):
        # The following data members are provided:
        # - _cell_nums and _node_nums;
        # - _dim;
        # - _cell_options;
        # - _folder;
        # - _parametric_shape_info: a list of (string, int);
        # - _node_boundary_info: a list of (int, float) or ((int, int, int), float) (2D) or
        #   ((int, int, int, int), float) (3D) that describes the boundary conditions.
        # - _interface_boundary_type: either 'no-slip' or 'free-slip' (default).

        # Sanity check the inputs.
        cell_nums = ndarray(cell_nums).astype(np.int32)
        assert cell_nums.size in [2, 3]

        # The value of E does not quite matter as it simply scale the QP problem by a constant factor.
        E = float(E)
        assert E > 0

        nu = float(nu)
        assert 0 < nu < 0.5

        vol_tol = float(vol_tol)
        assert 0 < vol_tol < 0.5

        edge_sample_num = int(edge_sample_num)
        assert edge_sample_num > 1

        if folder is not None:
            folder = Path(folder)
            create_folder(folder, exist_ok=False)

        # Create data members.
        self._cell_nums = np.copy(cell_nums)
        self._node_nums = ndarray([n + 1 for n in cell_nums]).astype(np.int32)
        self._dim = self._cell_nums.size
        self._cell_options = {
            'E': E, 'nu': nu, 'vol_tol': vol_tol, 'edge_sample_num': edge_sample_num
        }
        self._folder = folder

        ###########################################################################
        # Derived classes should implement these data members.
        ###########################################################################
        self._parametric_shape_info = []
        self._node_boundary_info = []
        self._interface_boundary_type = 'free-slip'

    ###########################################################################
    # Derived classes should implement these functions.
    ###########################################################################
    # Each scene must define the parametrization of the shape.
    # Input:
    # - x: a 1D array that stores the variables.
    # Output:
    # - param: a 1D array that directly specifies the shape parameters.
    # - jacobian: a matrix of size param.size x var.size.
    def _variables_to_shape_params(self, x):
        # By default, we have a trivial mapping.
        param = ndarray(x).ravel()
        return param, np.eye(x.size)

    # Each scene must present a loss and grad function defined on a velocity field.
    # Input:
    # - u: a 1D array that stores the whole velocity.
    # Output:
    # - loss: a scalar (float).
    # - grad: a 1D array that stores grad loss/grad u.
    def _loss_and_grad_on_velocity_field(self, u):
        raise NotImplementedError

    # Lower bounds, upper bounds, and initial configuration.
    def sample(self):
        raise NotImplementedError

    def lower_bound(self):
        raise NotImplementedError

    def upper_bound(self):
        raise NotImplementedError

    ###########################################################################
    # Other base-class functions.
    ###########################################################################
    # If u is an 1D array, reshape it to [*node_nums] x dim.
    # If it is a tensor of size [*node_nums] x dim, flatten it as a 1D array.
    def reshape_velocity_field(self, u):
        u_array = ndarray(u)
        assert len(u_array.shape) in [1, self._dim + 1]
        assert u_array.size == np.prod(self._node_nums) * self._dim
        if len(u_array.shape) == 1:
            return np.copy(u_array).reshape(tuple(self._node_nums) + (self._dim,))
        else:
            return np.copy(u_array).ravel()

    # The core of this class.
    # - x: shape parameters flattened into a 1D array.
    # - require_grad: whether to compute gradients or not.
    # - options:
    #   - 'solver': 'eigen' or 'pardiso'.
    def solve(self, x, require_grad, options):
        scene = Scene2d() if self._dim == 2 else Scene3d()

        # Initialize shapes.
        assert self._parametric_shape_info
        x = ndarray(x).copy().ravel()
        param, J = self._variables_to_shape_params(x)
        assert param.size == self.parameter_dim()
        cnt = 0
        names = []
        vals = []
        for k, v in self._parametric_shape_info:
            pk = param[cnt:cnt + v]
            cnt += v
            names.append(k)
            vals.append(pk)

        scene.InitializeShapeComposition([int(n) for n in self._cell_nums], names, vals)

        # Initialize cells.
        scene.InitializeCell(
            self._cell_options['E'],
            self._cell_options['nu'],
            self._cell_options['vol_tol'],
            self._cell_options['edge_sample_num']
        )

        # Initialize the Dirichlet boundary conditions.
        node_bnd_dict = {}
        for dof, val in self._node_boundary_info:
            dof = scene.GetNodeDof(dof[:-1], dof[-1])
            node_bnd_dict[dof] = val
        node_bnd_dofs = []
        node_bnd_vals = []
        for dof, val in node_bnd_dict.items():
            dof = int(dof)
            assert 0 <= dof < np.prod(self._node_nums)
            val = float(val)
            node_bnd_dofs.append(dof)
            node_bnd_vals.append(val)
        scene.InitializeDirichletBoundaryCondition(node_bnd_dofs, node_bnd_vals)

        # Initialize the interface boundary type.
        name_map = {
            'no-slip': 'no_slip',
            'free-slip': 'no_separation'
        }
        assert self._interface_boundary_type in name_map
        scene.InitializeBoundaryType(name_map[self._interface_boundary_type])

        # Solve the velocity.
        assert 'solver' in options and options['solver'] in ['eigen', 'pardiso']

        # Forward simulation to obtain the velocity field.
        solver = options['solver']
        forward_result = scene.Forward(solver)
        u = ndarray(scene.GetVelocityFieldFromForward(forward_result))

        info = { 'scene': scene, 'velocity_field': self.reshape_velocity_field(u) }
        loss, grad = self._loss_and_grad_on_velocity_field(u)
        if require_grad:
            # Backpropagation.
            grad = ndarray(scene.Backward(solver, forward_result, grad))
            grad = J.T @ grad
            return loss, grad, info
        else:
            return loss, info

    def render(self, xk, img_name, options):
        if self._dim == 2:
            self._render_2d(xk, img_name, options)
        else:
            self._render_3d(xk, image_name, options)

    def _render_2d(self, xk, img_name, options):
        assert self._folder
        loss, info = self.solve(xk, False, options)
        scene = info['scene']
        u_field = info['velocity_field']

        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        cx, cy = self._cell_nums
        padding = 3
        ax.set_xlim([-padding, cx + padding])
        ax.set_ylim([-padding, cy + padding])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Loss: {:3.6e}'.format(loss))

        # Plot cells.
        lines = []
        colors = []
        shift = 0.05
        fludic_node = np.ones((cx + 1, cy + 1))
        for i in range(cx):
            for j in range(cy):
                if scene.IsFluidCell((i, j)):
                    color = 'tab:blue'
                elif scene.IsSolidCell((i, j)):
                    color = 'tab:orange'
                    fludic_node[i, j] = fludic_node[i + 1, j] = fludic_node[i, j + 1] = fludic_node[i + 1, j + 1] = 0
                else:
                    color = 'tab:cyan'
                pts = [(i + shift, j + shift),
                    (i + 1 - shift, j + shift),
                    (i + 1 - shift, j + 1 - shift),
                    (i + shift, j + 1 - shift)
                ]
                lines += [
                    (pts[0], pts[1]),
                    (pts[1], pts[2]),
                    (pts[2], pts[3]),
                    (pts[3], pts[0])
                ]
                colors += [color,] * 4
        ax.add_collection(mc.LineCollection(lines, colors=colors, linestyle='-.', alpha=0.3))

        # Plot velocity fields.
        lines = []
        for i in range(cx + 1):
            for j in range(cy + 1):
                if not fludic_node[i, j]: continue
                v_begin = ndarray([i, j])
                v_end = v_begin + u_field[i, j]
                lines.append((v_begin, v_end))
        ax.add_collection(mc.LineCollection(lines, colors='tab:blue', linestyle='-'))

        # Plot solid-fluid interfaces.
        lines = []
        def cutoff(d0, d1):
            assert d0 * d1 <= 0
            # (0, d0), (t, 0), (1, d1).
            # t / -d0 = 1 / (d1 - d0)
            return -d0 / (d1 - d0)
        for i in range(cx):
            for j in range(cy):
                if not scene.IsMixedCell((i, j)): continue
                ps = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                ds = [scene.GetSignedDistance(p) for p in ps]
                ps = ndarray(ps)
                vs = []
                for k in range(4):
                    k_next = (k + 1) % 4
                    if ds[k] * ds[k_next] <= 0:
                        t = cutoff(ds[k], ds[k_next])
                        vs.append((1 - t) * ps[k] + t * ps[k_next])
                vs_len = len(vs)
                for k in range(vs_len):
                    lines.append((vs[k], vs[(k + 1) % vs_len]))
        ax.add_collection(mc.LineCollection(lines, colors='tab:olive', linestyle='-'))
        fig.savefig(self._folder / img_name)

    def _render_3d(self, xk, img_name, options):
        assert self._folder
        _, info = self.solve(xk, False, options)
        scene = info['scene']
        u_field = info['velocity_field']
        # TODO.

    def parameter_dim(self):
        return np.sum([v for _, v in self._parametric_shape_info])

    def cell_nums(self):
        return np.copy(self._cell_nums)

    def node_nums(self):
        return np.copy(self._node_nums)

    def dim(self):
        return self._dim

    def cell_options(self):
        return self._cell_options

    def folder(self):
        return self._folder

    def parametric_shape_info(self):
        return self._parametric_shape_info

    def node_boundary_info(self):
        return self._node_boundary_info

    def interface_boundary_type(self):
        return self._interface_boundary_type