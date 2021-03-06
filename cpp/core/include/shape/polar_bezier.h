#ifndef SHAPE_POLAR_BEZIER_H
#define SHAPE_POLAR_BEZIER_H

#include "shape/parametric_shape.h"
#include "shape/bezier.h"

// This class defines a Bezier curve in the polar coordinates.
// It is parametrized by the following values:
// - rho: a vector of real;
// - cx, cy: the center of the polar coordinates;
// - angle_offset: the initial angle offset of the polar coordinates.
// Just as before, we assume that t from 0 to 1 equals looping over the solid (positive) region in the
// counter-clockwise order.
class PolarBezier2d : public ParametricShape<2> {
public:
    PolarBezier2d(const bool flip);

    const real ComputeSignedDistanceAndGradients(const std::array<real, 2>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;

    bool flip_;
    int bezier_num_;
    std::vector<real> rho_;
    Vector2r center_;
    real angle_offset_;
    real angle_step_size_;

    // Derived data.
    std::vector<Vector2r> control_points_;
    std::vector<Matrix2Xr> control_points_gradients_;
    std::vector<std::shared_ptr<Bezier2d>> bezier_curves_;
};

// This class assumes a few PolarBezier2d defined on planes parallel to the xy plane.
// All PolarBezier2ds have aligned center and angle offset.
// The parameters are assumed to be organized as follows:
// - rho: 2 * bezier_num_ * z_level_num.
// - cx, cy.
// - angle_offset.
class PolarBezier3d : public ParametricShape<3> {
public:
    PolarBezier3d(const bool flip, const int z_level_num);

    const real ComputeSignedDistanceAndGradients(const std::array<real, 3>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;

    bool flip_;
    int bezier_num_;
    std::vector<std::vector<real>> rho_;
    Vector2r center_;
    real angle_offset_;
    real angle_step_size_;

    int z_level_num_;
    real dz_;

    // Derived data.
    // coeffs_ are used for interpolating curves vertically.
    MatrixXr coeffs_;
};

#endif