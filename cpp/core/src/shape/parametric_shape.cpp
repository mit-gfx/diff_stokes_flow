#include "shape/parametric_shape.h"
#include "common/common.h"

template<int dim>
ParametricShape<dim>::ParametricShape() : param_num_(0) {
    cell_num_prod_ = 1;
    node_num_prod_ = 1;
    for (int i = 0; i < dim; ++i) {
        cell_nums_[i] = 1;
        cell_num_prod_ *= cell_nums_[i];
        node_nums_[i] = cell_nums_[i] + 1;
        node_num_prod_ *= node_nums_[i];
    }
    signed_distances_ = std::vector<real>(node_num_prod_, 0);
    signed_distance_gradients_ = std::vector<std::vector<real>>(node_num_prod_, std::vector<real>(param_num_, 0));
    params_ = std::vector<real>();
}

template<int dim>
void ParametricShape<dim>::Initialize(const std::array<int, dim>& cell_nums, const std::vector<real>& params) {
    cell_nums_ = cell_nums;
    cell_num_prod_ = 1;
    node_num_prod_ = 1;
    for (int i = 0; i < dim; ++i) {
        cell_num_prod_ *= cell_nums_[i];
        node_nums_[i] = cell_nums_[i] + 1;
        node_num_prod_ *= node_nums_[i];
    }
    params_ = params;
    param_num_ = static_cast<int>(params.size());

    InitializeCustomizedData();

    // Now compute the gradients.
    signed_distances_.resize(node_num_prod_);
    signed_distance_gradients_.resize(node_num_prod_);
    for (int i = 0; i < node_num_prod_; ++i) signed_distance_gradients_[i].resize(param_num_);

    #pragma omp parallel for
    for (int i = 0; i < node_num_prod_; ++i) {
        const auto idx = GetIndex(i, node_nums_);
        // Cast to real.
        std::array<real, dim> p;
        for (int j = 0; j < dim; ++j) p[j] = static_cast<real>(idx[j]);
        signed_distances_[i] = ComputeSignedDistanceAndGradients(p, signed_distance_gradients_[i]);        
    }
}

template<int dim>
const int ParametricShape<dim>::cell_num(const int i) const {
    CheckError(0 <= i && i < dim, "Dimension out of bound.");
    return cell_nums_[i];
}

template<int dim>
const int ParametricShape<dim>::node_num(const int i) const {
    CheckError(0 <= i && i < dim, "Dimension out of bound.");
    return node_nums_[i];
}

template<int dim>
const real ParametricShape<dim>::signed_distance(const std::array<int, dim>& node_idx) const {
    return signed_distances_[GetIndex(node_idx, node_nums_)];
}

template<int dim>
const std::vector<real>& ParametricShape<dim>::signed_distance_gradients(
    const std::array<int, dim>& node_idx) const {
    return signed_distance_gradients_[GetIndex(node_idx, node_nums_)];
}

template class ParametricShape<2>;
template class ParametricShape<3>;