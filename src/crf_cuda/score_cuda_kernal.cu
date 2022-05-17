#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

namespace
{
    template <typename scalar_t>
    __global__ void init_score(const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
                               const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> tag,
                               const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
                               torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> score)
    {
        const int n = blockIdx.x;
        const int i = threadIdx.x;
        __shared__ float s[1024];
        if (i < lens[n])
        {
            s[i] = input[n][i][tag[n][i]];
        }
        else
        {
            s[i] = 0;
        }
        __syncthreads();
        for (int j = blockDim.x >> 1; j > 0; j >>= 1)
        {
            if (i < j)
                s[i] += s[i + j];
            __syncthreads();
        }
        if (i == 0)
            score[n] = s[0];
    }
    template <typename scalar_t>
    __global__ void sum_score(const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> weight,
                              const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> tag,
                              const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
                              const int64_t start_ix,
                              const int64_t end_ix,
                              torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> score)
    {
        const int n = blockIdx.x;
        const int i = threadIdx.x;
        __shared__ float s[1024];
        if (i == 0)
        {
            s[i] = weight[tag[n][i]][start_ix];
        }
        else if (i < lens[n])
        {
            s[i] = weight[tag[n][i]][tag[n][i - 1]];
        }
        else if (i == lens[n])
        {
            s[i] = weight[end_ix][tag[n][i - 1]];
        }
        else
        {
            s[i] = 0;
        }
        __syncthreads();
        for (int j = blockDim.x >> 1; j > 0; j >>= 1)
        {
            if (i < j)
                s[i] += s[i + j];
            __syncthreads();
        }
        if (i == 0)
            score[n] += s[0];
    }
    template <typename scalar_t>
    __global__ void init_score_backword(const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> score_grad,
                                        const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> tag,
                                        const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
                                        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input_grad)
    {
        const int n = blockIdx.x;
        const int i = threadIdx.x;
        if (i < lens[n])
        {
            input_grad[n][i][tag[n][i]] = score_grad[n];
        }
    }
    template <typename scalar_t>
    __global__ void sum_score_backword(const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> score_grad,
                                       const torch::PackedTensorAccessor<int64_t, 2, torch::RestrictPtrTraits, size_t> tag,
                                       const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
                                       const int64_t tag_size,
                                       const int64_t start_ix,
                                       const int64_t end_ix,
                                       torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> weight_grad)
    {
        const int n = blockIdx.x;
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        __shared__ int s[32][32];
        s[i][j] = 0;

        if (i == tag[n][0] && j == start_ix)
        {
            s[i][j] = 1;
        }
        else if (i == end_ix && j == tag[n][lens[n] - 1])
        {
            s[i][j] = 1;
        }
        else if (i != end_ix && j != end_ix)
        {
            for (int k = 1; k < lens[n]; k++)
            {
                if (tag[n][k] == i && tag[n][k - 1] == j)
                {
                    s[i][j] += 1;
                }
            }
        }
        weight_grad[n][i][j] = score_grad[n] * s[i][j];
    }
}

std::vector<torch::Tensor> score_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &length,
    const torch::Tensor &weight,
    const torch::Tensor &tags,
    const int64_t start_ix,
    const int64_t stop_ix)
{
    const int64_t batch_size = input.size(0);
    auto scores = torch::zeros({batch_size}, input.options());
    AT_DISPATCH_FLOATING_TYPES(input.type(), "score_forward", ([&]
                                                               { init_score<<<batch_size, 1024>>>(input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                                                  tags.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                                                  length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                                                  scores.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()); }));
    AT_DISPATCH_FLOATING_TYPES(input.type(), "score_forward", ([&]
                                                               { sum_score<<<batch_size, 1024>>>(weight.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                                                 tags.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                                                 length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                                                 start_ix,
                                                                                                 stop_ix,
                                                                                                 scores.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()); }));
    return {scores};
}

std::vector<torch::Tensor> score_cuda_backward(
    const torch::Tensor &score_grad,
    const torch::Tensor &tags,
    const int64_t tagsize,
    const int64_t start_idx,
    const int64_t end_idx,
    const torch::Tensor &length)
{
    const auto batch_size = score_grad.size(0);
    const auto max_len = tags.size(1);

    auto d_input = torch::zeros({batch_size, max_len, tagsize}, score_grad.options());
    auto d_weight = torch::zeros({batch_size, tagsize, tagsize}, score_grad.options());
    dim3 thread(tagsize, tagsize);
    AT_DISPATCH_FLOATING_TYPES(score_grad.type(), "score_backward", ([&]
                                                                     { init_score_backword<<<batch_size, 1024>>>(score_grad.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                                                                 tags.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                                                                 length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                                                                 d_input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()); }));
    AT_DISPATCH_FLOATING_TYPES(score_grad.type(), "score_backward", ([&]
                                                                     { sum_score_backword<<<batch_size, thread>>>(score_grad.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                                                                  tags.packed_accessor<int64_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                                                                  length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                                                                  tagsize,
                                                                                                                  start_idx,
                                                                                                                  end_idx,
                                                                                                                  d_weight.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()); }));
    const auto d_weight_t = d_weight.sum(0);
    return {d_input, d_weight_t};
}