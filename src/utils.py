import torch
def argmax(vec,dim=1):
    # return the argmax as a python int
    _, idx = torch.max(vec, dim)
    return idx.item()

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_batch(vecs):
    max_score_broadcast = torch.max(vecs, 1)[0].view(-1, 1)
    return max_score_broadcast + \
        torch.log(torch.sum(torch.exp(vecs - max_score_broadcast), 1)).view(-1, 1)
