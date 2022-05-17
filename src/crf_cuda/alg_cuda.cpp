#include <torch/extension.h>

#include <vector>
#include <iostream>
std::vector<torch::Tensor> alg_cuda_forward(
    torch::Tensor input,
    torch::Tensor length,
    torch::Tensor weights,
    int start_idx,
    int end_idx);

std::vector<torch::Tensor> alg_cuda_backward(
    torch::Tensor score_grad,
    torch::Tensor states,
    torch::Tensor stop_states,
    torch::Tensor length,
    int stop_idx);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> alg_forward(
    torch::Tensor input,
    torch::Tensor length,
    torch::Tensor weights,
    int start_idx,
    int end_idx)
{
    return alg_cuda_forward(input,length, weights, start_idx, end_idx);
}

std::vector<torch::Tensor> alg_backward(
    torch::Tensor score_grad,
    torch::Tensor states,
    torch::Tensor stop_states,
    torch::Tensor length,
    int stop_idx)
{
    return alg_cuda_backward(score_grad, states, stop_states,length, stop_idx);
}