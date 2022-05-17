import torch
from torch import nn
from torch.autograd import Function
from .crf_cuda import score_forward, score_backward

class ScoreCUDAFunction(Function):
    @staticmethod
    def forward(self, inputs, weights, tags, start_idx, end_idx, length):
        score, = score_forward(inputs, length, weights, tags, start_idx, end_idx)
        self.save_for_backward(tags, torch.scalar_tensor(weights.shape[0]),torch.scalar_tensor(start_idx),torch.scalar_tensor(end_idx),length)
        return score

    @staticmethod
    def backward(self, grad_output):
        d_input,d_weight = score_backward(grad_output, *self.saved_tensors)
        return d_input,d_weight,None,None,None,None

class Score(nn.Module):
    def __init__(self, state_size, start_idx, end_idx, adjacency_matrix=None):
        super(Score, self).__init__()
        self.state_size = state_size
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.weights = nn.Parameter(torch.Tensor(state_size, state_size))
        if adjacency_matrix is None:
            self.adjacency_matrix = torch.zeros_like(self.weights)
            self.adjacency_matrix[self.start_idx, :] = 1.
            self.adjacency_matrix[:, self.end_idx] = 1.
        else:
            self.adjacency_matrix = adjacency_matrix
        self.reset_parameters()
    
    def reset_parameters(self):
        self.weights.data.uniform_(-1.0 / self.state_size, 1.0 / self.state_size)
        self.weights.data =self.weights.data - 1000*self.adjacency_matrix

    def _score_sentence(self, feats, tags):
        score = feats[torch.arange(len(tags),dtype=torch.long),tags].sum()
        tags = torch.cat([torch.tensor([self.start_idx], dtype=torch.long), tags, torch.tensor([self.end_idx], dtype=torch.long)])
        score += self.weights[tags[1:], tags[:-1]].sum()
        return score

    def forward(self, feats, tags, length):
        if self.weights.is_cuda:
            return ScoreCUDAFunction.apply(feats, self.weights, tags, self.start_idx, self.end_idx, length)
        else:
            out = []
            for i,j,k in zip(feats,tags,length):
                out.append(self._score_sentence(i[:k],j[:k]))
            return torch.stack(out)