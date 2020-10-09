#include "scene/scene.h"
#include "common/common.h"

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
        const bool consistency = dirichlet_conditions_.find(i) == dirichlet_conditions_.end()
            || dirichlet_conditions_.at(i) == 0;
        if (!consistency) {
            std::cout << "Dof " << i << " should have been a solid node but was given nonzero velocity: "
                << dirichlet_conditions_.at(i) << "." << std::endl;
            const auto node_idx = GetIndex(i / dim, shape_.node_nums());
            std::cout << "Node coordinates:";
            for (int k = 0; k < dim; ++k) std::cout << " " << node_idx[k];
            std::cout << std::endl;
            CheckError(false, "Inconsistent Dirichlet conditions.");
        }
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
const std::vector<real> Scene<dim>::Forward(const std::string& qp_solver_name) {
    CheckError(!cells_.empty() && shape_.param_num(), "You must call all initialization function first.");
    // Now assemble the QP problem.
    // min 0.5 * u * K * u
    // s.t. C * u = d
    //        u_i = u_i*

    // Assemble K, C, and d.
    // TODO: Use OpenMP to parallelize the code?
    const int cell_num = shape_.cell_num_prod();
    SparseMatrixElements K_nonzeros, C_nonzeros;
    std::vector<real> d_vec;
    const int param_num = shape_.param_num();
    std::vector<SparseMatrixElements> dK_nonzeros(param_num), dC_nonzeros(param_num);
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
        // Assemble dK.
        for (int p = 0; p < param_num; ++p) {
            MatrixXr dK = K;
            dK.setZero();
            for (int j = 0; j < cell.corner_num_prod(); ++j) {
                const auto corner_idx = GetIndex(j, cell.corner_nums());
                std::array<int, dim> node_idx = cell_idx;
                for (int k = 0; k < dim; ++k) node_idx[k] += corner_idx[k];
                dK += cell.energy_matrix_gradients()[j] * shape_.signed_distance_gradients(node_idx)[p];
            }
            for (int ii = 0; ii < dof_map_size; ++ii)
                for (int jj = 0; jj < dof_map_size; ++jj)
                    dK_nonzeros[p].push_back(Eigen::Triplet<real>(dof_map[ii], dof_map[jj], dK(ii, jj)));
        }

        // Assemble C.
        if (cell.IsFluidCell()) continue;
        const VectorXr& c = cell.dirichlet_vector();
        std::vector<VectorXr> dc(param_num, VectorXr::Zero(c.size()));
        for (int p = 0; p < param_num; ++p) {
            for (int j = 0; j < cell.corner_num_prod(); ++j) {
                const auto corner_idx = GetIndex(j, cell.corner_nums());
                std::array<int, dim> node_idx = cell_idx;
                for (int k = 0; k < dim; ++k) node_idx[k] += corner_idx[k];
                dc[p] += cell.dirichlet_vector_gradients().col(j) * shape_.signed_distance_gradients(node_idx)[p];
            }
        }
        if (boundary_type_ == BoundaryType::NoSlip) {
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < cell.corner_num_prod(); ++k) {
                    C_nonzeros.push_back(Eigen::Triplet<real>(C_row_num, dof_map[k * dim + j], c(k)));
                    for (int p = 0; p < param_num; ++p) {
                        dC_nonzeros[p].push_back(Eigen::Triplet<real>(C_row_num, dof_map[k * dim + j], dc[p](k)));
                    }
                }
                d_vec.push_back(0);
                ++C_row_num;
            }
        } else if (boundary_type_ == BoundaryType::NoSeparation) {
            const Eigen::Matrix<real, dim, 1>& normal = cell.normal();
            std::vector<Eigen::Matrix<real, dim, 1>> dnormal(param_num, Eigen::Matrix<real, dim, 1>::Zero());
            for (int p = 0; p < param_num; ++p) {
                for (int j = 0; j < cell.corner_num_prod(); ++j) {
                    const auto corner_idx = GetIndex(j, cell.corner_nums());
                    std::array<int, dim> node_idx = cell_idx;
                    for (int k = 0; k < dim; ++k) node_idx[k] += corner_idx[k];
                    dnormal[p] += cell.normal_gradients().col(j) * shape_.signed_distance_gradients(node_idx)[p];
                }
            }
            for (int j = 0; j < dim; ++j) {
                for (int k = 0; k < cell.corner_num_prod(); ++k) {
                    C_nonzeros.push_back(Eigen::Triplet<real>(C_row_num, dof_map[k * dim + j], c(k) * normal(j)));
                    for (int p = 0; p < param_num; ++p) {
                        dC_nonzeros[p].push_back(Eigen::Triplet<real>(C_row_num, dof_map[k * dim + j],
                            dc[p](k) * normal(j) + c(k) * dnormal[p](j)
                        ));
                    }
                }
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
    KC_ = ToSparseMatrix(var_num + C_row_num, var_num + C_row_num, KC_nonzeros);
    VectorXr d_ext = VectorXr::Zero(var_num + C_row_num);
    d_ext.tail(C_row_num) = d;

    dKC_nonzeros_ = dK_nonzeros;
    for (int p = 0; p < param_num; ++p) {
        for (const auto& triplet : dC_nonzeros[p]) {
            const int row = triplet.row();
            const int col = triplet.col();
            const real val = triplet.value();
            dKC_nonzeros_[p].push_back(Eigen::Triplet<real>(var_num + row, col, val));
            dKC_nonzeros_[p].push_back(Eigen::Triplet<real>(col, var_num + row, val));
        }
    }

    // Solve KC * x = d_ext and compute dloss_dparams.
    // KC * x = d_ext.
    // dKC * x + KC * dx = 0.
    // KC * dx = -dKC * x.
    // dx = -KC^(-1) * (dKC * x).
    // dloss_dparams[i] = -dloss_du * (KC^(-1) * (dKC * x))[:var_num].
    // So, here is our solution to d_loss_d_params:
    // - Append 0 to dloss_du so that it has the length (var_num + C_row_num).
    // - Solve for KC * y = -dloss_du. y = -KC^(-1) * dloss_du.
    // - For each parameter index i, compute dloss_dparams[i] = y.dot(dKC[i] * x).
    if (boundary_type_ != BoundaryType::NoSlip && boundary_type_ != BoundaryType::NoSeparation) {
        PrintError("You must implement delta d_ext since you are using a new boundary type.");
    }
    VectorXr x = VectorXr::Zero(var_num + C_row_num);
    if (qp_solver_name == "eigen") {
        eigen_solver_.compute(KC_);
        CheckError(eigen_solver_.info() == Eigen::Success, "SparseLU fails to compute the sparse matrix: "
            + std::to_string(eigen_solver_.info()));
        x = eigen_solver_.solve(d_ext);
        CheckError(eigen_solver_.info() == Eigen::Success, "SparseLU fails to solve d_ext: "
            + std::to_string(eigen_solver_.info()));
    } else if (qp_solver_name == "pardiso") {
        pardiso_solver_.Compute(KC_);
        x = pardiso_solver_.Solve(d_ext);
    } else {
        PrintError("Unsupported QP solver: " + qp_solver_name + ". Please use eigen or pardiso.");
    }
    // Sanity check.
    const real abs_tol = ToReal(1e-4);
    const real rel_tol = ToReal(1e-3);
    const real abs_error_x = (KC_ * x - d_ext).norm();
    CheckError(abs_error_x <= d_ext.norm() * rel_tol + abs_tol, "QP solver fails to solve d_ext.");

    // Return the solution.
    return ToStdVector(x);
}

template<int dim>
const std::vector<real> Scene<dim>::Backward(const std::string& qp_solver_name,
    const std::vector<real>& forward_result,
    const std::vector<real>& partial_loss_partial_solution_field) {
    // Obtain dimension information.
    const int var_num = shape_.node_num_prod() * dim;
    const int C_row_num = static_cast<int>(forward_result.size()) - var_num;

    // - For each parameter index i, compute dloss_dparams[i] = y.dot(dKC[i] * x).
    VectorXr dloss_du = VectorXr::Zero(var_num + C_row_num);
    CheckError(static_cast<int>(partial_loss_partial_solution_field.size()) == var_num,
        "Inconsistent length of partial_loss_partial_solution_field.");
    for (int i = 0; i < var_num; ++i) dloss_du(i) = partial_loss_partial_solution_field[i];
    VectorXr y = VectorXr::Zero(var_num + C_row_num);
    if (qp_solver_name == "eigen") {
        y = eigen_solver_.solve(-dloss_du);
        CheckError(eigen_solver_.info() == Eigen::Success, "SparseLU fails to solve dloss_du: "
            + std::to_string(eigen_solver_.info()));
    } else if (qp_solver_name == "pardiso") {
        y = pardiso_solver_.Solve(-dloss_du);
    } else {
        PrintError("Unsupported QP solver: " + qp_solver_name + ". Please use eigen or pardiso.");
    }

    const real abs_tol = ToReal(1e-4);
    const real rel_tol = ToReal(1e-3);
    const real abs_error_y = (KC_ * y + dloss_du).norm();
    CheckError(abs_error_y <= dloss_du.norm() * rel_tol + abs_tol, "QP solver fails to solve dloss_du.");

    const int param_num = shape_.param_num();
    std::vector<real> d_loss_d_params(param_num, 0);
    #pragma omp parallel for
    for (int i = 0; i < param_num; ++i) {
        for (const auto& triplet : dKC_nonzeros_[i]) {
            d_loss_d_params[i] += y(triplet.row()) * triplet.value() * forward_result[triplet.col()];
        }
    }
    return d_loss_d_params;
}

template<int dim>
const std::vector<real> Scene<dim>::GetVelocityFieldFromForward(const std::vector<real>& forward_result) const {
    const int var_num = shape_.node_num_prod() * dim;
    return std::vector<real>(forward_result.data(), forward_result.data() + var_num);
}

template<int dim>
const int Scene<dim>::GetNodeDof(const std::array<int, dim>& node_idx, const int node_dim) const {
    return GetIndex(node_idx, shape_.node_nums()) * dim + node_dim;
}

template<int dim>
const real Scene<dim>::GetSignedDistance(const std::array<int, dim>& node_idx) const {
    return shape_.signed_distance(node_idx);
}

template<int dim>
const std::vector<real> Scene<dim>::GetSignedDistanceGradients(const std::array<int, dim>& node_idx) const {
    return shape_.signed_distance_gradients(node_idx);
}

template<int dim>
const bool Scene<dim>::IsSolidCell(const std::array<int, dim>& cell_idx) const {
    const auto& cell = cells_.at(GetIndex(cell_idx, shape_.cell_nums()));
    return cell.IsSolidCell();
}

template<int dim>
const bool Scene<dim>::IsFluidCell(const std::array<int, dim>& cell_idx) const {
    const auto& cell = cells_.at(GetIndex(cell_idx, shape_.cell_nums()));
    return cell.IsFluidCell();
}

template<int dim>
const bool Scene<dim>::IsMixedCell(const std::array<int, dim>& cell_idx) const {
    const auto& cell = cells_.at(GetIndex(cell_idx, shape_.cell_nums()));
    return cell.IsMixedCell();
}

template class Scene<2>;
template class Scene<3>;