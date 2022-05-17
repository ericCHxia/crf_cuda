from crf_cuda import CRF
import torch
torch.manual_seed(0)

def test_crf():
    tag_size = 6
    feats = torch.randn((2,10,6))
    tags = torch.randint(0,3,(2,10))
    lens = torch.tensor([10,2]).long()

    model_cpu = CRF(tag_size,4,5)
    cpu_loss = model_cpu.neg_log_likelihood(feats,tags,lens)
    cpu_loss = cpu_loss.mean()
    cpu_loss.backward()

    cpu_loss = cpu_loss.clone()

    cpu_grad = model_cpu.weights.grad.clone()

    model_cpu.zero_grad()
    model_cpu.zero_grad()

    feats = feats.cuda()
    tags = tags.cuda()
    lens = lens.cuda()

    model_cuda = model_cpu.cuda()

    cuda_loss = model_cuda.neg_log_likelihood(feats,tags,lens)
    cuda_loss = cuda_loss.mean()
    cuda_loss.backward()

    cuda_loss = cuda_loss.cpu()
    cuda_grad = model_cuda.weights.grad.cpu()

    assert torch.allclose(cpu_grad,cuda_grad,atol=1e-6)
    assert torch.allclose(cpu_loss,cuda_loss)