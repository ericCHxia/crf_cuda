import torch
from torch import nn
from torch.autograd import Function

from .crf_cuda import alg_backward, alg_forward
from .utils import log_sum_exp_batch,log_sum_exp

class ALGCUDAFunction(Function):
    @staticmethod
    def forward(self, inputs, weights, start_idx, end_idx, length):
        score, state, stop_state = alg_forward(inputs, length, weights, start_idx, end_idx);
        self.save_for_backward(state, stop_state, length, torch.scalar_tensor(end_idx))
        return score
    
    @staticmethod
    def backward(self, grad_output):
        d_input,d_weight = alg_backward(grad_output, *self.saved_tensors)
        with torch.no_grad():
            d_input = d_input*grad_output[:,None,None]
            d_weight = d_weight*(grad_output.mean())
        return d_input,d_weight,None,None,None

class ALG(nn.Module):
    def __init__(self, state_size, start_idx, end_idx, adjacency_matrix=None):
        super(ALG, self).__init__()
        self.state_size = state_size
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.weights = nn.Parameter(torch.Tensor(state_size, state_size))
        if adjacency_matrix is None:
            self.adjacency_matrix = torch.zeros_like(self.weights)
            self.adjacency_matrix[self.start_idx, :] = 1.
            self.adjacency_matrix[:, self.end_idx] = 1.
        else:
            self.adjacency_matrix = 1.0 - adjacency_matrix
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weights.data.uniform_(-1.0 / self.state_size, 1.0 / self.state_size)
        self.weights.data =self.weights.data - 10000*self.adjacency_matrix.to(self.weights.device)

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.state_size), -10000.)
        init_alphas[0][self.start_idx] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = forward_var + self.weights + feat.view(-1, 1)
            forward_var = log_sum_exp_batch(alphas_t).view(1, -1)
        terminal_var = forward_var + self.weights[self.end_idx]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def forward(self, text, lens):
        if self.weights.device != torch.device('cpu'):
            return ALGCUDAFunction.apply(text, self.weights, self.start_idx, self.end_idx, lens)
        else:
            result = []
            for inputs,l in zip(text,lens):
                result.append(self._forward_alg(inputs[:l]))
            return torch.stack(result)
