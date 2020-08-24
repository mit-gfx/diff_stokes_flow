#ifndef SCENE_SCENE_H
#define SCENE_SCENE_H

#include "common/config.h"
#include "shape/shape_composition.h"
#include "cell/cell.h"

template<int dim>
class Scene {
public:
    Scene();

    void InitializeShapeComposition(const std::array<int, dim>& cell_nums, const std::vector<std::string>& shape_names,
        const std::vector<std::vector<real>>& shape_params);
    void InitializeCell(const real E, const real nu, const real threshold, const int edge_sample_num);
    void InitializeDirichletBoundaryCondition(const std::vector<int>& dofs, const std::vector<real>& values);
    void InitializeBoundaryType(const std::string& boundary_type);
    const std::vector<real> Solve(const std::string& qp_solver_name) const;

    const int GetNodeDof(const std::array<int, dim>& node_idx, const int node_dim) const;
    const real GetSignedDistance(const std::array<int, dim>& node_idx) const;

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
};

#endif