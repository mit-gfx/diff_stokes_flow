import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from skimage import measure

from py_diff_stokes_flow.core.py_diff_stokes_flow_core import StdRealVector, Scene2d, Scene3d
from py_diff_stokes_flow.common.common import ndarray, create_folder, print_warning
from py_diff_stokes_flow.common.renderer import PbrtRenderer
from py_diff_stokes_flow.common.project_path import root_path

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
#     def _color_velocity(self, u):
#         return 0.5
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
            create_folder(folder, exist_ok=True)

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
    # Either (param, jacobian) or a list of (param, jacobian).
    # - param: a 1D array that directly specifies the shape parameters.
    # - jacobian: a matrix of size param.size x var.size.
    def _variables_to_shape_params(self, x):
        # By default, we have a trivial mapping.
        param = ndarray(x).ravel()
        return param, np.eye(x.size)

    # Each scene must present a loss and grad function defined on a velocity field.
    # Input:
    # - u: if the scene has only one mode, u is a 1D array that stores the whole velocity.
    #      Otherwise, u is a list of 1D arrays whose lengths = mode number.
    # Output:
    # Either loss and grad or a list of (loss, grad), depending on the number of modes.
    # - loss: a scalar (float).
    # - grad: a 1D array that stores grad loss/grad u.
    def _loss_and_grad_on_velocity_field(self, u):
        raise NotImplementedError

    # This function is used by _render_3d to determine the color of the velocity field.
    # Input:
    # - u: a 3D array.
    # Output:
    # - intercept: a floating point number between 0 and 1.
    def _color_velocity(self, u):
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
        # Initialize shapes.
        assert self._parametric_shape_info
        x = ndarray(x).copy().ravel()
        # Determine the mode number.
        param_and_J = self._variables_to_shape_params(x)
        if len(param_and_J) == 2 and isinstance(param_and_J[0], np.ndarray):
            param_and_J = [param_and_J,]
        mode_num = len(param_and_J)
        scenes = [Scene2d() if self._dim == 2 else Scene3d() for _ in range(mode_num)]

        # Solve the velocity.
        assert 'solver' in options and options['solver'] in ['eigen', 'pardiso']
        # Forward simulation to obtain the velocity field.
        solver = options['solver']
        info = []
        u = []
        forward_result = []

        for (param, _), scene in zip(param_and_J, scenes):
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
                dof = [int(d) for d in dof]
                dof = scene.GetNodeDof(dof[:-1], dof[-1])
                node_bnd_dict[dof] = val
            node_bnd_dofs = []
            node_bnd_vals = []
            for dof, val in node_bnd_dict.items():
                dof = int(dof)
                assert 0 <= dof < np.prod(self._node_nums) * self._dim
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

            # Solve for the velocity field.
            forward_result_single = scene.Forward(solver)
            u_single = ndarray(scene.GetVelocityFieldFromForward(forward_result_single))
            info_single = { 'scene': scene, 'velocity_field': self.reshape_velocity_field(u_single) }
            info.append(info_single)
            u.append(u_single)
            forward_result.append(forward_result_single)

        # Compute the loss.
        loss_and_grad = self._loss_and_grad_on_velocity_field(u[0] if mode_num == 1 else u)
        if mode_num == 1:
            loss_and_grad = [loss_and_grad,]
        loss = np.sum([l for l, _ in loss_and_grad])

        if not require_grad:
            return loss, info

        # Compute the gradients.
        grads = [g for _, g in loss_and_grad]
        grad = 0
        for (_, J), scene, g, f in zip(param_and_J, scenes, grads, forward_result):
            g = ndarray(scene.Backward(solver, f, g))
            grad += J.T @ g
        return loss, grad, info

    def render(self, xk, img_name, options):
        if self._dim == 2:
            self._render_2d(xk, img_name, options)
        else:
            self._render_3d(xk, img_name, options)

    def _render_2d(self, xk, img_name, options):
        assert self._folder
        loss, info = self.solve(xk, False, options)
        # For the basic _render_2d function, we assume mode = 1.
        mode_num = len(info)
        for m in range(mode_num):
            mode_folder = Path(self._folder / 'mode_{:04d}'.format(m))
            create_folder(mode_folder, exist_ok=True)
            scene = info[m]['scene']
            u_field = info[m]['velocity_field']

            cx, cy = self._cell_nums
            face_color = ndarray([247 / 255, 247 / 255, 247 / 255])
            plt.rcParams['figure.facecolor'] = face_color
            plt.rcParams['axes.facecolor'] = face_color
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            padding = 5
            ax.set_title('Loss: {:3.6e}'.format(loss))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([-padding, cx + padding])
            ax.set_ylim([-padding, cy + padding])
            ax.set_aspect('equal')
            ax.axis('off')

            # Plot cells.
            lines = []
            colors = []
            shift = 0.0
            fluidic_node = np.ones((cx + 1, cy + 1))
            for i in range(cx):
                for j in range(cy):
                    if scene.IsFluidCell((i, j)):
                        color = 'k'
                    elif scene.IsSolidCell((i, j)):
                        color = 'k'
                        fluidic_node[i, j] = fluidic_node[i + 1, j] = fluidic_node[i, j + 1] = fluidic_node[i + 1, j + 1] = 0
                    else:
                        color = 'k'
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
            ax.add_collection(mc.LineCollection(lines, colors=colors, linewidth=0.5))

            # Plot velocity fields.
            cmap = plt.get_cmap('coolwarm')
            lines = []
            colors = []
            u_min = np.inf
            u_max = -np.inf
            for i in range(cx + 1):
                for j in range(cy + 1):
                    uij = u_field[i, j]
                    uij_norm = np.linalg.norm(uij)
                    if uij_norm > 0:
                        if uij_norm > u_max:
                            u_max = uij_norm
                        if uij_norm < u_min:
                            u_min = uij_norm

            for i in range(cx + 1):
                for j in range(cy + 1):
                    if not fluidic_node[i, j]: continue
                    uij = u_field[i, j]
                    uij_norm = np.linalg.norm(uij)
                    v0 = ndarray([i, j])
                    v1 = v0 + uij
                    lines.append((v0, v1))
                    # Determine the color.
                    color = cmap((uij_norm - u_min) / (u_max - u_min))
                    colors.append(color)

            ax.add_collection(mc.LineCollection(lines, colors=colors, linewidth=1.0))

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
            ax.add_collection(mc.LineCollection(lines, colors='tab:orange', linewidth=1))

            # Plot other customized data if needed.
            self._render_customized_2d(scene, ax)

            fig.savefig(mode_folder / img_name)
            plt.close()

    def _render_customized_2d(self, scene, ax):
        pass

    def _render_3d(self, xk, img_name, options):
        assert self._folder
        _, info = self.solve(xk, False, options)
        # For the basic _render_2d function, we assume mode = 1.
        mode_num = len(info)
        for m in range(mode_num):
            mode_folder = Path(self._folder / 'mode_{:04d}'.format(m))
            create_folder(mode_folder, exist_ok=True)
            scene = info[m]['scene']
            u_field = info[m]['velocity_field']

            # A proper range for placing the scene is [-0.4, 0.4] x [-0.3, 0.3] x [0, 0.6].
            render_options = {
                'file_name': str(mode_folder / img_name),
                'resolution': (1024, 768),
                'light_map': 'uffizi-large.exr',
                'light_map_scale': 0.75,
                'sample': options['spp'],
                'max_depth': 2,
                'camera_pos': (0.1, -0.65, 1.1),
                'camera_lookat': (0, 0, 0),
            }
            renderer = PbrtRenderer(render_options)
            renderer.add_tri_mesh(Path(root_path) / 'asset/mesh/plane.obj',
                transforms=[('s', 1.5)],
                color=(.4, .4, .4), texture_img='background.png')

            # Render the solid-fluid interface.
            cx, cy, cz = self._cell_nums
            nx, ny, nz = self._node_nums
            # How to use cmap:
            # cmap(0.0) to cmap(1.0) covers the whole range of the colormap.
            cmap = plt.get_cmap('jet')
            scale = np.min([0.8 / cx, 0.6 / cy, 0.6 / cz])
            transforms=[
                ('t', (-cx / 2, -cy / 2, 0)),
                ('s', scale)
            ]
            # Assemble an obj mesh.
            image_prefix = '.'.join(img_name.split('.')[:-1])
            interface_file_name = mode_folder / '{}.obj'.format(image_prefix)
            sdf = np.zeros((nx, ny, nz))
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        sdf[i, j, k] = scene.GetSignedDistance((i, j, k))
            sdf = ndarray(sdf)
            verts, faces, _, _ = measure.marching_cubes_lewiner(sdf, 0)
            verts = ndarray(verts)
            faces = ndarray(faces).astype(np.int32) + 1

            # Write obj files.
            with open(interface_file_name, 'w') as f:
                for v in verts:
                    f.write('v {:6f} {:6f} {:6f}\n'.format(*v))
                for fi in faces:
                    f.write('f {:d} {:d} {:d}\n'.format(*fi))

            renderer.add_tri_mesh(interface_file_name, transforms=transforms,
                color=(1.0, 0.61, 0.0))

            # Render the velocity field.
            lines = []
            max_u_len = -np.inf
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if scene.GetSignedDistance((i, j, k)) >= 0: continue
                        v_begin = ndarray([i, j, k])
                        v_end = v_begin + u_field[i, j, k]
                        u_len = np.linalg.norm(u_field[i, j, k])
                        if u_len > max_u_len:
                            max_u_len = u_len
                        lines.append((v_begin, v_end))
            # Scale the line lengths so that the maximum length is 1/10 of the longest axis.
            lines_scale = 0.1 * np.max(self._cell_nums) / max_u_len
            # width is 1/10 of the cell size.
            width = 0.1
            for v_begin, v_end in lines:
                # Compute the color.
                color_idx = self._color_velocity(v_end - v_begin)
                color = ndarray(cmap(color_idx))[:3]
                v0 = v_begin
                v3 = (v_end - v_begin) * lines_scale + v_begin
                v1 = (2 * v0 + v3) / 3
                v2 = (v0 + 2 * v3) / 3
                renderer.add_shape_mesh({
                        'name': 'curve',
                        'point': ndarray([v0, v1, v2, v3]),
                        'width': width
                    },
                    color=color,
                    transforms=transforms
                )
            renderer.render(verbose=True)

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
