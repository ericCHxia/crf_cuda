#include <torch/extension.h>

#include <vector>
#include <iostream>

std::vector<torch::Tensor> score_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& length,
    const torch::Tensor& weight,
    const torch::Tensor& tags,
    const int64_t start_ix,
    const int64_t stop_ix
);

std::vector<torch::Tensor> score_cuda_backward(
    const torch::Tensor &score_grad,
    const torch::Tensor &tags,
    const int64_t tagsize,
    const int64_t start_idx,
    const int64_t end_idx,
    const torch::Tensor &length);

std::vector<torch::Tensor> score_forward(
    const torch::Tensor& input,
    const torch::Tensor& length,
    const torch::Tensor& weight,
    const torch::Tensor& tags,
    const int64_t start_ix,
    const int64_t stop_ix
){
    return score_cuda_forward(input, length, weight, tags, start_ix, stop_ix);
};

std::vector<torch::Tensor> score_backward(
    const torch::Tensor &score_grad,
    const torch::Tensor &tags,
    const int64_t tagsize,
    const int64_t start_idx,
    const int64_t end_idx,
    const torch::Tensor &length
){
    return score_cuda_backward(score_grad, tags, tagsize, start_idx, end_idx, length);
};