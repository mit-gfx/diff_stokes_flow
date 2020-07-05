#ifndef SHAPE_PARAMETRIC_SHAPE_H
#define SHAPE_PARAMETRIC_SHAPE_H

#include "common/config.h"

// This class implements a parametric shape in a regular grid whose lower left corner is assumed to be (0, 0) in 2D or
// (0, 0, 0) in 3D. The cell size is assumed to be 1. Each parametric shape induces a signed distance function f(p)
// where p is a 2D or 3D point. f(p) >= 0 represents the boundary and interior of the shape. Moreover, we assume solid
// (fluid) regions have positive (negative) distances to the surface of the shapes.
template<int dim>
class ParametricShape {
public:
    ParametricShape();
    virtual ~ParametricShape() {}

    void Initialize(const std::array<int, dim>& cell_nums, const std::vector<real>& params);

    const int cell_num(const int i) const;
    const int node_num(const int i) const;
    const int param_num() const { return param_num_; }
    const std::vector<real>& params() const { return params_; }

    const std::vector<real>& signed_distances() const { return signed_distances_; }
    const real signed_distance(const std::array<int, dim>& node_idx) const;

    virtual void Backward(const std::vector<real>& dl_dsigned_distances, std::vector<real>& dl_dparams) const;

protected:
    virtual void ComputeSignedDistances();
    const int GetIndex(const std::array<int, dim>& node_idx) const;
    const std::array<int, dim> GetIndex(const int node_idx) const;

    const Eigen::Matrix<int, dim, 1> ToEigenIndex(const std::array<int, dim>& node_idx) const;
    const std::array<int, dim> ToStdIndex(const Eigen::Matrix<int, dim, 1>& node_idx) const;

private:
    int cell_num_prod_;
    std::array<int, dim> cell_nums_;
    int node_num_prod_;
    std::array<int, dim> node_nums_;
    int param_num_;
    std::vector<real> params_;

    std::vector<real> signed_distances_;
};

#endif