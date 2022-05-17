#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>
namespace
{
    template <typename scalar_t>
    __global__ void log_sum_exp_cuda_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> input,
        const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
        const int t,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> state,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output)
    {
        const int n = blockIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;

        if (c >= input.size(1) || lens[n] <= t)
            return;

        auto max_score = input[n][c][0];
        auto max_idx = 0;
        for (int i = 1; i < input.size(2); ++i)
        {
            if (input[n][c][i] > max_score)
            {
                max_score = input[n][c][i];
                max_idx = i;
            }
        }

        auto sum_exp = 0.0;
        for (int i = 0; i < input.size(2); ++i)
        {
            state[n][t][c][i] = exp(input[n][c][i] - max_score);
            sum_exp += state[n][t][c][i];
        }
        for (int i = 0; i < input.size(2); i++)
        {
            state[n][t][c][i] /= sum_exp;
        }
        output[n][c] = max_score + log(sum_exp);
    }
    template <typename scalar_t>
    __global__ void alg_forward_add(const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
                                    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> weight,
                                    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> forward_var,
                                    const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
                                    const size_t t,
                                    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output)
    {
        const int n = blockIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (lens[n] > t && c < weight.size(1) * weight.size(0))
        {
            const int i = c % weight.size(1);
            const int j = c / weight.size(1);

            output[n][i][j] = forward_var[n][j] + weight[i][j] + feats[n][t][i];
        }
    }

    template <typename scalar_t>
    __global__ void alg_forward_add_end(const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> weight,
                                        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> forward_var,
                                        const int end_idx,
                                        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output)
    {
        const int n = blockIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < weight.size(1))
        {
            output[n][c] = forward_var[n][c] + weight[end_idx][c];
        }
    }

    template <typename scalar_t>
    __global__ void alg_backword_add(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> d_forward_var,
        const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
        torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> states,
        const int t)
    {
        const int n = blockIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (lens[n] > t && c < states.size(2) * states.size(3))
        {
            const int i = c % states.size(3);
            const int j = c / states.size(3);
            states[n][t][i][j] = d_forward_var[n][i] * states[n][t][i][j];
        }
    }

    template <typename scalar_t>
    __global__ void add_with_length(
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> states,
        const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
        const int t,
        torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> output)
    {
        const int n = blockIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (lens[n] > t && c < states.size(2) * states.size(3))
        {
            const int i = c % states.size(3);
            const int j = c / states.size(3);
            output[n][i][j] = output[n][i][j] + states[n][t][i][j];
        }
    }

    template <typename scalar_t>
    __global__ void states_sum(
        const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> states,
        const torch::PackedTensorAccessor<int64_t, 1, torch::RestrictPtrTraits, size_t> lens,
        const int t,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output)
    {
        const int n = blockIdx.y;
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (lens[n] > t && c < states.size(3))
        {
            output[n][c] = 0;
            for (int i = 0; i < states.size(2); ++i)
            {
                output[n][c] += states[n][t][i][c];
            }
        }
    }
}

std::vector<torch::Tensor> alg_cuda_forward(
    torch::Tensor input,
    torch::Tensor length,
    torch::Tensor weights,
    int start_idx,
    int end_idx)
{
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int state_size = input.size(2);

    auto forward_add = torch::zeros({batch_size, state_size, state_size}, input.options());
    auto state = torch::zeros({batch_size, seq_len, state_size, state_size}, input.options());
    auto stop_state = torch::zeros({batch_size, 1, state_size}, input.options());
    auto score = torch::zeros({batch_size, 1}, input.options());

    auto forward_var = torch::full({batch_size, state_size}, -10000, input.options());

    for (size_t i = 0; i < forward_var.size(0); ++i)
    {
        forward_var[i][start_idx] = 0;
    }

    const int threads = 1024;
    const dim3 blocks_add((state_size * state_size + threads - 1) / threads, batch_size);
    const dim3 blocks_log_sum_exp((state_size + threads - 1) / threads, batch_size);

    for (size_t t = 0; t < seq_len; ++t)
    {
        AT_DISPATCH_FLOATING_TYPES(forward_add.type(), "alg_forward", ([&]
                                                                       { alg_forward_add<scalar_t><<<blocks_add, threads>>>(
                                                                             input.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                             weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                             forward_var.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                             length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                             t,
                                                                             forward_add.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()); }));
        AT_DISPATCH_FLOATING_TYPES(forward_add.type(), "alg_forward", ([&]
                                                                       { log_sum_exp_cuda_kernel<scalar_t><<<blocks_log_sum_exp, threads>>>(
                                                                             forward_add.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                             length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                             t,
                                                                             state.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                             forward_var.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));
    }

    auto forward_add_end = forward_add.select(1, end_idx);

    AT_DISPATCH_FLOATING_TYPES(forward_add.type(), "alg_forward", ([&]
                                                                   { alg_forward_add_end<scalar_t><<<blocks_log_sum_exp, threads>>>(
                                                                         weights.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                         forward_var.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                         end_idx,
                                                                         forward_add_end.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));

    stop_state = stop_state.view({batch_size, 1, 1, state_size});
    auto forward_add_end_view = forward_add_end.view({batch_size, 1, state_size});

    AT_DISPATCH_FLOATING_TYPES(forward_add.type(), "alg_forward", ([&]
                                                                   { log_sum_exp_cuda_kernel<scalar_t><<<blocks_log_sum_exp, threads>>>(
                                                                         forward_add_end_view.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                                                                         length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                         0,
                                                                         stop_state.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                         score.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));
    stop_state = stop_state.view({batch_size, state_size});
    score = score.view({batch_size});
    return {score, state, stop_state};
}

std::vector<torch::Tensor> alg_cuda_backward(
    torch::Tensor score_grad,
    torch::Tensor states,
    torch::Tensor stop_states,
    torch::Tensor length,
    int stop_idx)
{
    const int batch_size = states.size(0);
    const int seq_len = states.size(1);
    const int state_size = states.size(2);

    auto d_weight = torch::zeros({batch_size, state_size, state_size}, states.options());
    auto d_forward_var = stop_states.clone();
    d_weight.select(1, stop_idx).copy_(stop_states);
    const int threads = 1024;
    const dim3 blocks_add((state_size * state_size + threads - 1) / threads, batch_size);
    const dim3 blocks_log_sum_exp((state_size + threads - 1) / threads, batch_size);

    for (int t = seq_len - 1; t >= 0; --t)
    {
        AT_DISPATCH_FLOATING_TYPES(stop_states.type(), "alg_backward", ([&]
                                                                        { alg_backword_add<scalar_t><<<blocks_add, threads>>>(
                                                                              d_forward_var.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                                                                              length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                              states.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                              t); }));
        AT_DISPATCH_FLOATING_TYPES(stop_states.type(), "alg_backward", ([&]
                                                                        { add_with_length<scalar_t><<<blocks_add, threads>>>(
                                                                              states.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                              length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                              t,
                                                                              d_weight.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()); }));

        AT_DISPATCH_FLOATING_TYPES(stop_states.type(), "alg_backward", ([&]
                                                                        { states_sum<scalar_t><<<blocks_log_sum_exp, threads>>>(
                                                                              states.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                                                                              length.packed_accessor<int64_t, 1, torch::RestrictPtrTraits, size_t>(),
                                                                              t,
                                                                              d_forward_var.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()); }));
    }

    auto d_input = states.sum(3);
    auto d_weight_mean = d_weight.sum(0);
    return {d_input, d_weight_mean};
}