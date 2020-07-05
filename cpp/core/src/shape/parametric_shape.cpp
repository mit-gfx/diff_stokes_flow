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
    signed_distances_ = std::vector<real>(node_num_prod_, 0);
    ComputeSignedDistances();
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
    return signed_distances_[GetIndex(node_idx)];
}

template<int dim>
void ParametricShape<dim>::Backward(const std::vector<real>& dl_dsigned_distances, std::vector<real>& dl_dparams)
    const {
    dl_dparams.resize(param_num_, 0);
    std::fill(dl_dparams.begin(), dl_dparams.end(), 0);
}

template<int dim>
void ParametricShape<dim>::ComputeSignedDistances() {
    std::fill(signed_distances_.begin(), signed_distances_.end(), 0);
}

template<int dim>
const int ParametricShape<dim>::GetIndex(const std::array<int, dim>& node_idx) const {
    for (int i = 0; i < dim; ++i) CheckError(0 <= node_idx[i] && node_idx[i] < node_nums_[i],
        "Node index out of bound.");
    int node_idx_linear = node_idx[0];
    for (int i = 0; i < dim - 1; ++i) {
        node_idx_linear *= node_nums_[i + 1];
        node_idx_linear += node_idx[i + 1];
    }
    return node_idx_linear;
}

template<int dim>
const std::array<int, dim> ParametricShape<dim>::GetIndex(const int node_idx) const {
    CheckError(0 <= node_idx && node_idx < node_num_prod_, "Node index out of bound.");
    std::array<int, dim> node_idx_array;
    node_idx_array[dim - 1] = node_idx % node_nums_[dim - 1];
    int node_idx_copy = node_idx / node_nums_[dim - 1];
    for (int i = dim - 2; i >= 0; --i) {
        node_idx_array[i] = node_idx_copy % node_nums_[i];
        node_idx_copy /= node_nums_[i];
    }
    return node_idx_array;
}

template<int dim>
const Eigen::Matrix<int, dim, 1> ParametricShape<dim>::ToEigenIndex(const std::array<int, dim>& node_idx) const {
    Eigen::Matrix<int, dim, 1> node_idx_eig;
    for (int i = 0; i < dim; ++i) {
        node_idx_eig(i) = node_idx[i];
    }
    return node_idx_eig;
}

template<int dim>
const std::array<int, dim> ParametricShape<dim>::ToStdIndex(const Eigen::Matrix<int, dim, 1>& node_idx) const {
    std::array<int, dim> node_idx_std;
    for (int i = 0; i < dim; ++i) {
        node_idx_std[i] = node_idx(i);
    }
    return node_idx_std;
}

template class ParametricShape<2>;
template class ParametricShape<3>;