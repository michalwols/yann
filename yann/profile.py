from torch.autograd.profiler import profile


def param_count(model):
  return sum(p.numel() for p in model.parameters())
