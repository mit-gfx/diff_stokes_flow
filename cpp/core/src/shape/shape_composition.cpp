#include "shape/shape_composition.h"
#include "common/common.h"
#include "shape/bezier.h"
#include "shape/plane.h"
#include "shape/sphere.h"
#include "shape/polar_bezier.h"

template<>
void ShapeComposition<2>::AddParametricShape(const std::string& name, const int param_num) {
    ParametricShapeInfo<2> info;
    info.name = name;
    info.param_begin_idx = 0;
    info.param_num = param_num;
    if (name == "bezier") {
        info.shape = std::make_shared<Bezier2d>();
    } else if (name == "plane") {
        info.shape = std::make_shared<Plane<2>>();
    } else if (name == "sphere") {
        info.shape = std::make_shared<Sphere<2>>();
    } else if (name == "polar_bezier") {
        // Do not flip in 2D.
        const bool flip = false;
        info.shape = std::make_shared<PolarBezier2d>(flip);
    } else {
        PrintError("Unsupported shape name: " + name);
    }
    shape_info_.push_back(info);
}

template<>
void ShapeComposition<3>::AddParametricShape(const std::string& name, const int param_num) {
    ParametricShapeInfo<3> info;
    info.name = name;
    info.param_begin_idx = 0;
    info.param_num = param_num;
    if (name == "bezier") {
        info.shape = std::make_shared<Bezier3d>();
    } else if (name == "plane") {
        info.shape = std::make_shared<Plane<3>>();
    } else if (name == "sphere") {
        info.shape = std::make_shared<Sphere<3>>();
    } else if (BeginsWith(name, "polar_bezier")) {
        // Fetch z_level_num.
        const int z_level_num_signed = std::stoi(name.substr(std::string("polar_bezier").size()));
        const bool flip = z_level_num_signed < 0;
        const int z_level_num = flip ? -z_level_num_signed : z_level_num_signed;
        info.shape = std::make_shared<PolarBezier3d>(flip, z_level_num);
    } else {
        PrintError("Unsupported shape name: " + name);
    }
    shape_info_.push_back(info);
}

template<int dim>
void ShapeComposition<dim>::InitializeCustomizedData() {
    int param_cur_idx = 0;
    for (auto& info : shape_info_) {
        info.param_begin_idx = param_cur_idx;
        const std::vector<real> shape_param(ParametricShape<dim>::params().begin() + param_cur_idx,
            ParametricShape<dim>::params().begin() + param_cur_idx + info.param_num);
        info.shape->Initialize(ParametricShape<dim>::cell_nums(), shape_param);
        param_cur_idx += info.param_num;
    }
}

template<int dim>
const real ShapeComposition<dim>::ComputeSignedDistanceAndGradients(const std::array<real, dim>& point,
    std::vector<real>& grad) const {
    real min_pos_dist = std::numeric_limits<real>::infinity();
    real min_neg_dist = std::numeric_limits<real>::infinity();
    std::vector<real> min_pos_dist_grad, min_neg_dist_grad;
    CheckError(!shape_info_.empty(), "You need to have at least one shape.");
    int min_pos_param_begin = 0, min_neg_param_begin = 0;
    int min_pos_param_num = 0, min_neg_param_num = 0;
    bool is_solid = false;
    for (const auto& info : shape_info_) {
        std::vector<real> shape_grad;
        const real shape_dist = info.shape->ComputeSignedDistanceAndGradients(point, shape_grad);
        if (shape_dist >= 0) {
            // This point is inside the solid phase.
            is_solid = true;
            if (shape_dist < min_pos_dist) {
                min_pos_dist = shape_dist;
                min_pos_dist_grad = shape_grad;
                min_pos_param_begin = info.param_begin_idx;
                min_pos_param_num = info.param_num;
            }
        } else {
            // This point is in the fluid phase.
            if (-shape_dist < min_neg_dist) {
                min_neg_dist = -shape_dist;
                min_neg_dist_grad = shape_grad;
                min_neg_param_begin = info.param_begin_idx;
                min_neg_param_num = info.param_num;
            }
        }
    }
    grad.clear();
    grad.resize(ParametricShape<dim>::param_num(), 0);
    if (is_solid) {
        for (int i = 0; i < min_pos_param_num; ++i)
            grad[min_pos_param_begin + i] = min_pos_dist_grad[i];
        return min_pos_dist;
    } else {
        for (int i = 0; i < min_neg_param_num; ++i)
            grad[min_neg_param_begin + i] = min_neg_dist_grad[i];
        return -min_neg_dist;
    }
}

template class ParametricShapeInfo<2>;
template class ParametricShapeInfo<3>;
template class ShapeComposition<2>;
template class ShapeComposition<3>;