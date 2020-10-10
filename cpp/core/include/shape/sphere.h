#ifndef SHAPE_SPHERE_H
#define SHAPE_SPHERE_H

#include "shape/parametric_shape.h"

// A sphere is defined by its center c (2D or 3D) and radius r.
// Positive distance = solid region.
template<int dim>
class Sphere : public ParametricShape<dim> {
public:
    const real ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;

    // Sphere equation: dist = r - |x - c|
    Eigen::Matrix<real, dim, 1> center_;
    real radius_;
};

#endif