#ifndef SHAPE_PLANE_H
#define SHAPE_PLANE_H

#include "shape/parametric_shape.h"

// A plane is defined as ax + by + c = 0 (2D) or ax + by + cz + d = 0 (3D).
// ax + by + c >= 0 is the solid region.
template<int dim>
class Plane : public ParametricShape<dim> {
public:
    const real ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;

    // Plane equation: normal_.dot(x) + offset_ >= 0 <-> the solid area.
    Eigen::Matrix<real, dim, 1> normal_;
    real offset_;
};

#endif