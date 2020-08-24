#include "scene/scene.h"
#include "common/common.h"
#include "Eigen/SparseLU"

template<int dim>
Scene<dim>::Scene() : boundary_type_(BoundaryType::NoSeparation) {}

template<int dim>
void Scene<dim>::InitializeShapeComposition(const std::array<int, dim>& cell_nums,
    const std::vector<std::string>& shape_names, const std::vector<std::vector<real>>& shape_params) {
    CheckError(shape_names.size() == shape_params.size(), "Inconsistent shape names and parameters.");
    shape_.Clear();

    const int shape_num = static_cast<int>(shape_names.size());
    std::vector<real> params;
    for (int i = 0; i < shape_num; ++i) {
        shape_.AddParametricShape(shape_names[i], static_cast<int>(shape_params[i].size()));
        params.insert(params.end(), shape_params[i].begin(), shape_params[i].end());
    }
    shape_.Initialize(cell_nums, params);
}

template<int dim>
void Scene<dim>::InitializeCell(const real E, const real nu, const real threshold, const int edge_sample_num) {
    CheckError(shape_.param_num(), "You must call InitializeShapeComposition first.");
    cells_.clear();
    const int cell_num_prod = shape_.cell_num_prod();
    cells_.resize(cell_num_prod);
    #pragma omp parallel for
    for (int i = 0; i < cell_num_prod; ++i) {
        Cell<dim>& cell = cells_[i];
        const auto cell_idx = GetIndex(i, shape_.cell_nums());
        std::vector<real> sdf_at_corners(cell.corner_num_prod());
        for (int j = 0; j < cell.corner_num_prod(); ++j) {
            const auto corner_idx = GetIndex(j, cell.corner_nums());
            std::array<int, dim> node_idx = cell_idx;
            for (int k = 0; k < dim; ++k) node_idx[k] += corner_idx[k];
            sdf_at_corners[j] = shape_.signed_distance(node_idx);
        }
        // Ready to initialize this cell.
        cell.Initialize(E, nu, threshold, edge_sample_num, sdf_at_corners);
    }
}

template<int dim>
void Scene<dim>::InitializeDirichletBoundaryCondition(const std::vector<int>& dofs, const std::vector<real>& values) {
    CheckError(dofs.size() == values.size(), "Inconsistent dofs and values");
    CheckError(!cells_.empty(), "You must call InitializeCell first.");
    dirichlet_conditions_.clear();

    const int dofs_num = static_cast<int>(dofs.size());
    for (int i = 0; i < dofs_num; ++i)
        dirichlet_conditions_[dofs[i]] = values[i];

    // For nodes that are not adjacent to fluid or mixed cells, velocities are fixed to 0.
    std::vector<bool> free_dofs(shape_.node_num_prod() * dim, false);
    for (int i = 0; i < shape_.cell_num_prod(); ++i) {
        const auto& cell = cells_[i];
        if (cell.IsSolidCell()) continue;
        const auto cell_idx = GetIndex(i, shape_.cell_nums());
        for (int j = 0; j < cell.corner_num_prod(); ++j) {
            const auto corner_idx = GetIndex(j, cell.corner_nums());
            std::array<int, dim> node_idx = cell_idx;
            for (int k = 0; k < dim; ++k) node_idx[k] += corner_idx[k];
            const int idx = GetIndex(node_idx, shape_.node_nums());
            for (int k = 0; k < dim; ++k) free_dofs[idx * dim + k] = true;
        }
    }
    for (int i = 0; i < shape_.node_num_prod() * dim; ++i) {
        if (free_dofs[i]) continue;
        CheckError(dirichlet_conditions_.find(i) == dirichlet_conditions_.end()
            || dirichlet_conditions_.at(i) == 0, "Inconsistent Dirichlet conditions.");
        dirichlet_conditions_[i] = 0;
    }
}

template<int dim>
void Scene<dim>::InitializeBoundaryType(const std::string& boundary_type) {
    if (boundary_type == "no_slip") boundary_type_ = BoundaryType::NoSlip;
    else if (boundary_type == "no_separation") boundary_type_ = BoundaryType::NoSeparation;
    else PrintError("Unsupported boundary type: " + boundary_type);
}

template<int dim>
const std::vector<real> Scene<dim>::Solve(const std::string& qp_solver_name) const {
    CheckError(!cells_.empty() && shape_.param_num(), "You must calll all initialization function first.");
    // Now assemble the QP problem.
    // min 0.5 * u * K * u
    // s.t. C * u = d
    //        u_i = u_i*

    // Assemble K, C, and d.
    // TODO: Use OpenMP to parallelize the code?
    const int cell_num = shape_.cell_num_prod();
    SparseMatrixElements K_nonzeros, C_nonzeros;
    std::vector<real> d_vec;
    int C_row_num = 0;
    for (int i = 0; i < cell_num; ++i) {
        const auto& cell = cells_[i];
        if (cell.IsSolidCell()) continue;

        // Remap node indices from this cell to the grid.
        const auto cell_idx = GetIndex(i, shape_.cell_nums());
        std::vector<int> dof_map;
        const int dof_map_size = cell.corner_num_prod() * dim;
        dof_map.reserve(dof_map_size);
        for (int j = 0; j < cell.corner_num_prod(); ++j) {
            const auto corner_idx = GetIndex(j, cell.corner_nums());
            std::array<int, dim> node_idx = cell_idx;
            for (int k = 0; k < dim; ++k) node_idx[k] += corner_idx[k];
            const int idx = GetIndex(node_idx, shape_.node_nums());
            for (int k = 0; k < dim; ++k) dof_map.push_back(idx * dim + k);
        }

        // Assemble K.
        const MatrixXr& K = cell.energy_matrix();
        for (int ii = 0; ii < dof_map_size; ++ii)
            for (int jj = 0; jj < dof_map_size; ++jj)
                K_nonzeros.push_back(Eigen::Triplet<real>(dof_map[ii], dof_map[jj], K(ii, jj)));

        // Assemble C.
        if (cell.IsFluidCell()) continue;
        const VectorXr& c = cell.dirichlet_vector();
        if (boundary_type_ == BoundaryType::NoSlip) {
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < cell.corner_num_prod(); ++k)
                    C_nonzeros.push_back(Eigen::Triplet<real>(C_row_num, dof_map[k * dim + j], c(k)));
                d_vec.push_back(0);
                ++C_row_num;
            }
        } else if (boundary_type_ == BoundaryType::NoSeparation) {
            const Eigen::Matrix<real, dim, 1>& normal = cell.normal();
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < cell.corner_num_prod(); ++k)
                    C_nonzeros.push_back(Eigen::Triplet<real>(C_row_num, dof_map[k * dim + j], c(k) * normal(j)));
            }
            d_vec.push_back(0);
            ++C_row_num;
        } else PrintError("Unsupported boundary type.");
    }

    // Enforce Dirichlet boundary conditions.
    for (const auto& pair : dirichlet_conditions_) {
        C_nonzeros.push_back(Eigen::Triplet<real>(C_row_num, pair.first, 1));
        d_vec.push_back(pair.second);
        ++C_row_num;
    }

    const int var_num = shape_.node_num_prod() * dim;
    // For our reference, here are the definitions of K and C.
    // const SparseMatrix K = ToSparseMatrix(var_num, var_num, K_nonzeros);
    // const SparseMatrix C = ToSparseMatrix(C_row_num, var_num, C_nonzeros);
    const VectorXr d = Eigen::Map<const VectorXr>(d_vec.data(), d_vec.size());

    // Solve QP problem:
    //      min 0.5 * u * K * u
    //      s.t. C * u = d.
    // The KKT system:
    //      K * u + C' * la = 0
    //      C * u = d
    //      [K, C'] * [u ] = [0]
    //      [C,  0]   [la]   [d]
    SparseMatrixElements KC_nonzeros = K_nonzeros;
    for (const auto& triplet : C_nonzeros) {
        const int row = triplet.row();
        const int col = triplet.col();
        const real val = triplet.value();
        KC_nonzeros.push_back(Eigen::Triplet<real>(var_num + row, col, val));
        KC_nonzeros.push_back(Eigen::Triplet<real>(col, var_num + row, val));
    }
    const SparseMatrix KC = ToSparseMatrix(var_num + C_row_num, var_num + C_row_num, KC_nonzeros);
    VectorXr d_ext = VectorXr::Zero(var_num + C_row_num);
    d_ext.tail(C_row_num) = d;
    // Solve KC * x = d_ext.
    VectorXr x = VectorXr::Zero(var_num + C_row_num);
    if (qp_solver_name == "eigen") {
        Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(KC);
        CheckError(solver.info() == Eigen::Success, "SparseLU fails to compute the sparse matrix: "
            + std::to_string(solver.info()));
        x = solver.solve(d_ext);
        CheckError(solver.info() == Eigen::Success, "SparseLU fails to solve the right-hand vector: "
            + std::to_string(solver.info()));
    } else if (qp_solver_name == "pardiso") {
        // TODO: Add Pardiso.
        PrintError("Pardiso has not been included.");
    } else {
        PrintError("Unsupported QP solver: " + qp_solver_name + ". Please use eigen or pardiso.");
    }
    // Sanity check.
    const real abs_error = (KC * x - d_ext).norm();
    const real abs_tol = ToReal(1e-4);
    const real rel_tol = ToReal(1e-3);
    CheckError(abs_error <= d_ext.norm() * rel_tol + abs_tol, "QP solver fails.");

    // Return the solution.
    const VectorXr u = x.head(var_num);
    return std::vector<real>(u.data(), u.data() + var_num);
}

template<int dim>
const int Scene<dim>::GetNodeDof(const std::array<int, dim>& node_idx, const int node_dim) const {
    return GetIndex(node_idx, shape_.node_nums()) * dim + node_dim;
}

template<int dim>
const real Scene<dim>::GetSignedDistance(const std::array<int, dim>& node_idx) const {
    return shape_.signed_distance(node_idx);
}

template class Scene<2>;
template class Scene<3>;