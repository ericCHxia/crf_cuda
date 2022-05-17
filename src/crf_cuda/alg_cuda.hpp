#ifndef ALG_CUDA_H
#define ALG_CUDA_H
#include <torch/extension.h>

#include <vector>
#include <iostream>

std::vector<torch::Tensor> alg_forward(
    torch::Tensor input,
    torch::Tensor length,
    torch::Tensor weights,
    int start_idx,
    int end_idx);

std::vector<torch::Tensor> alg_backward(
    torch::Tensor score_grad,
    torch::Tensor states,
    torch::Tensor stop_states,
    torch::Tensor length,
    int stop_idx);
#endif