#ifndef SCORE_CUDA_H
#define SCORE_CUDA_H
#include <torch/extension.h>

#include <vector>
#include <iostream>

std::vector<torch::Tensor> score_forward(
    const torch::Tensor& input,
    const torch::Tensor& length,
    const torch::Tensor& weight,
    const torch::Tensor& tags,
    const int64_t start_ix,
    const int64_t stop_ix
);
#endif