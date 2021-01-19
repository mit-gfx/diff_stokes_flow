#ifndef SCENE_SCENE_H
#define SCENE_SCENE_H

#include "common/config.h"
#include "shape/shape_composition.h"
#include "cell/cell.h"
#include "Eigen/SparseLU"
#include "solver/pardiso_solver.h"

template<int dim>
class Scene {
public:
    Scene();

    void InitializeShapeComposition(const std::array<int, dim>& cell_nums, const std::vector<std::string>& shape_names,
        const std::vector<std::vector<real>>& shape_params);
    void InitializeCell(const real E, const real nu, const real threshold, const int edge_sample_num);
    void InitializeDirichletBoundaryCondition(const std::vector<int>& dofs, const std::vector<real>& values);
    void InitializeBoundaryType(const std::string& boundary_type);

    const std::vector<real> Forward(const std::string& qp_solver_name);
    const std::vector<real> Backward(const std::string& qp_solver_name, const std::vector<real>& forward_result,
        const std::vector<real>& partial_loss_partial_solution_field);
    const std::vector<real> GetVelocityFieldFromForward(const std::vector<real>& forward_result) const;

    const int GetNodeDof(const std::array<int, dim>& node_idx, const int node_dim) const;
    const real GetSignedDistance(const std::array<int, dim>& node_idx) const;
    const std::vector<real> GetSignedDistanceGradients(const std::array<int, dim>& node_idx) const;

    const std::array<real, dim> GetFluidicForceDensity(const std::array<int, dim>& node_idx) const;

    const bool IsSolidCell(const std::array<int, dim>& cell_idx) const;
    const bool IsFluidCell(const std::array<int, dim>& cell_idx) const;
    const bool IsMixedCell(const std::array<int, dim>& cell_idx) const;

    // Recall that normal.dot(x) + offset >= 0 is the solid phase in the cell.
    const std::array<real, dim> GetNormalInMixedCell(const std::array<int, dim>& cell_idx) const;
    const real GetOffsetInMixedCell(const std::array<int, dim>& cell_idx) const;

    // Some special energy types.
    // E = 0.5 * u * K * u.
    const real ComputeElasticEnergy(const std::vector<real>& velocity_field) const;
    const std::vector<real> ComputeElasticEnergyVelocityGradient(const std::vector<real>& velocity_field) const;
    const std::vector<real> ComputeElasticEnergyParameterGradient(const std::vector<real>& velocity_field) const;

private:
    // Geometry information.
    ShapeComposition<dim> shape_;
    // Cell information.
    std::vector<Cell<dim>> cells_;
    // Dirichlet boundary conditions.
    std::map<int, real> dirichlet_conditions_;
    // Boundary type.
    enum BoundaryType { NoSlip, NoSeparation };
    BoundaryType boundary_type_;

    // Data members shared by Forward and Backward.
    Eigen::SparseLU<SparseMatrix, Eigen::COLAMDOrdering<int>> eigen_solver_;
    PardisoSolver pardiso_solver_;
    SparseMatrix KC_;
    std::vector<SparseMatrixElements> dKC_nonzeros_;

    // Fluidic force density: this implements -Ku.
    std::vector<real> fluidic_force_density_;
};

#endif