import torch
import torch.nn as nn

from .alg import ALGCUDAFunction
from .score import ScoreCUDAFunction
from .utils import log_sum_exp, log_sum_exp_batch,argmax


class CRF(nn.Module):
    def __init__(self, tag_size, start_idx, end_idx, adjacency_matrix=None):
        super(CRF, self).__init__()
        self.tag_size = tag_size
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.weights = nn.Parameter(torch.Tensor(tag_size, tag_size))
        if adjacency_matrix is None:
            self.adjacency_matrix = torch.zeros_like(self.weights)
            self.adjacency_matrix[self.start_idx, :] = 1.
            self.adjacency_matrix[:, self.end_idx] = 1.
        else:
            self.adjacency_matrix = 1. - adjacency_matrix
        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.uniform_(-1.0 / self.tag_size, 1.0 / self.tag_size)
        self.weights.data =self.weights.data - 10000*self.adjacency_matrix.to(self.weights.device)
    
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tag_size), -10000.)
        init_alphas[0][self.start_idx] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = forward_var + self.weights + feat.view(-1, 1)
            forward_var = log_sum_exp_batch(alphas_t).view(1, -1)
        terminal_var = forward_var + self.weights[self.end_idx]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _score_sentence(self, feats, tags):
        score = feats[torch.arange(len(tags),dtype=torch.long),tags].sum()
        tags = torch.cat([torch.tensor([self.start_idx], dtype=torch.long), tags, torch.tensor([self.end_idx], dtype=torch.long)])
        score += self.weights[tags[1:], tags[:-1]].sum()
        return score

    def neg_log_likelihood(self, feats, tags, lens):
        if self.weights.is_cuda:
            forward_score = ALGCUDAFunction.apply(feats, self.weights, self.start_idx, self.end_idx, lens)
            gold_score = ScoreCUDAFunction.apply(feats, self.weights, tags, self.start_idx, self.end_idx, lens)
        else:
            forward_score = []
            gold_score = []
            for feat, tag, l in zip(feats, tags, lens):
                forward_score.append(self._forward_alg(feat[:l]))
                gold_score.append(self._score_sentence(feat[:l], tag[:l]))
            forward_score = torch.stack(forward_score)
            gold_score = torch.stack(gold_score)
        return forward_score - gold_score
    
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_size), -10000.).to(feats.device)
        init_vvars[0][self.start_idx] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tag_size):
                next_tag_var = forward_var + self.weights[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.weights[self.end_idx]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.start_idx
        best_path.reverse()
        return path_score, best_path

    def forward(self, feats, lens):
        path_scores = []
        best_paths = []
        for feat, l in zip(feats, lens):
            score, best_path = self._viterbi_decode(feat[:l])
            path_scores.append(score)
            best_paths.append(best_path)
        return path_scores, best_paths