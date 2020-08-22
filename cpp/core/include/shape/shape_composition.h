#ifndef SHAPE_SHAPE_COMPOSITION_H
#define SHAPE_SHAPE_COMPOSITION_H

#include "shape/parametric_shape.h"

template<int dim>
struct ParametricShapeInfo {
public:
    std::string name;
    std::shared_ptr<ParametricShape<dim>> shape;
    int param_begin_idx;
    int param_num;
};

// The solid region in this ShapeComposition class is the union of all parametric shapes.
template<int dim>
class ShapeComposition : public ParametricShape<dim> {
public:
    void AddParametricShape(const std::string& name, const int param_num);

    const real ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
        std::vector<real>& grad) const override;

private:
    void InitializeCustomizedData() override;

    std::vector<ParametricShapeInfo<dim>> shape_info_;
};

#endif