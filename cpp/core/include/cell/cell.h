#ifndef CELL_CELL_H
#define CELL_CELL_H

#include "common/config.h"

// Cell coordinates: lower left: (0, 0). Upper right: (1, 1).
template<int dim>
class Cell {
public:
    Cell() {}

    void Initialize(const real E, const real nu, const real threshold, const int edge_sample_num,
        const std::vector<real>& sdf_at_corners);

    const Eigen::Matrix<real, dim, 1>& normal() const { return normal_; }
    const real offset() const { return offset_; }
    const real sample_area(const int sample_idx) const;
    const real sample_area(const std::array<int, dim>& sample_idx) const;
    const std::vector<real>& sample_areas() const { return sample_areas_; }
    const real sample_boundary_area(const int sample_idx) const;
    const real sample_boundary_area(const std::array<int, dim>& sample_idx) const;
    const std::vector<real>& sample_boundary_areas() const { return sample_boundary_areas_; }
    const real area() const { return area_; }
    const MatrixXr& energy_matrix() const { return energy_matrix_; }
    const VectorXr& dirichlet_vector() const { return dirichlet_vector_; }

    const bool IsSolidCell() const { return area_ <= threshold_; }
    const bool IsFluidCell() const { return area_ >= 1 - threshold_; }
    const bool IsMixedCell() const { return !IsSolidCell() && !IsFluidCell(); }

private:
    const Eigen::Matrix<real, dim + 1, 1> FitBoundary(const std::vector<real>& sdf_at_corners) const;
    void ComputeSampleAreaAndBoundaryArea(const int sample_idx, real& area, real& boundary_area) const;
    const MatrixXr ComputeEnergyMatrix() const;
    const MatrixXr VelocityToDeformationGradient(const Eigen::Matrix<real, dim, 1>& material_coordinates) const;
    const VectorXr ComputeDirichletVector() const;

    int corner_num_prod_;   // 4 in 2D and 8 in 3D.
    std::array<int, dim> corner_nums_;   // (2, 2) in 2D and (2, 2, 2) in 3D.

    // Material parameters.
    real E_;
    real nu_;
    real la_;
    real mu_;

    // \Sum sample_areas_ <= threshold: Solid.
    // \Sum sample_areas_ >= 1 - threshold: Fluid.
    // Otherwise: mixed.
    real threshold_;

    // normal_.dot(x) + offset_ >= 0 is the solid phase in the cell.
    Eigen::Matrix<real, dim, 1> normal_;
    real offset_;

    // We divide each edge into edge_sample_num_ bins and place samples at the center of each subcell.
    int edge_sample_num_;
    int sample_num_prod_;
    std::array<int, dim> sample_nums_;
    // This is the fluid area inside each subcell. len(sample_areas_) = pow(edge_sample_num_, dim).
    std::vector<real> sample_areas_;
    // For 2D, sample_boundary_areas_ is the length of the boundary inside the subcell.
    // For 3D, it is the area of the boundary inside the subcell.
    std::vector<real> sample_boundary_areas_;
    // area_ = \Sum sample_areas_.
    real area_;

    // Energy quadratic term.
    // Let u be the flattened velocity (8-D in 2D and 24-D in 3D). The elastic energy is defined as:
    //      E = 0.5 * u * energy_matrix_ * u.
    // Essentially, energy_matrix is the negation of the stiffness matrix.
    MatrixXr energy_matrix_;
    // Dirichlet boundary conditions.
    // Let u be 2 x 4 (2D) or 3 x 8 (3D) matrix representing the velocity at each corner. We will use
    //      u[i].dot(dirichlet_vector_)
    // to represent the integral of u[i] over the boundary region.
    VectorXr dirichlet_vector_;
};

#endif