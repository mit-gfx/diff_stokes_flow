#ifndef SHAPE_PARAMETRIC_SHAPE_H
#define SHAPE_PARAMETRIC_SHAPE_H

#include "common/config.h"

// This class implements a parametric shape in a regular grid whose lower left corner is assumed to be (0, 0) in 2D or
// (0, 0, 0) in 3D. The cell size is assumed to be 1. Each parametric shape induces a signed distance function f(p)
// where p is a 2D or 3D point. f(p) >= 0 represents the boundary and interior of the shape. Moreover, we assume solid
// (fluid) regions have positive (negative) distances to the surface of the shapes.
// If the boundary is needed to be classified into solid/fluid phases, we will classify it into the solid phase.
template<int dim>
class ParametricShape {
public:
    ParametricShape();
    virtual ~ParametricShape() {}

    void Initialize(const std::array<int, dim>& cell_nums, const std::vector<real>& params);

    const int cell_num(const int i) const;
    const std::array<int, dim>& cell_nums() const { return cell_nums_; }
    const int node_num(const int i) const;
    const int node_num_prod() const { return node_num_prod_; }
    const int param_num() const { return param_num_; }
    const std::vector<real>& params() const { return params_; }

    const std::vector<real>& signed_distances() const { return signed_distances_; }
    const real signed_distance(const std::array<int, dim>& node_idx) const;
    const std::vector<real>& signed_distance_gradient(const std::array<int, dim>& node_idx) const;

    virtual const real ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
        std::vector<real>& grad) const = 0;

protected:
    // Give derived class a chance to initialize customized data.
    virtual void InitializeCustomizedData() {}

    const int GetIndex(const std::array<int, dim>& node_idx) const;
    const std::array<int, dim> GetIndex(const int node_idx) const;

private:
    int cell_num_prod_; // = \Pi cell_nums_.
    std::array<int, dim> cell_nums_;
    int node_num_prod_; // = \Pi numde_nums_.
    std::array<int, dim> node_nums_;
    int param_num_; // = len(params_).
    std::vector<real> params_;

    // signed_distances_[node_idx].
    std::vector<real> signed_distances_;
    // signed_distance_gradients_[node_idx][param_idx].
    std::vector<std::vector<real>> signed_distance_gradients_;
};

#endif