from torch.autograd.profiler import profile
# TODO:
# - https://github.com/pytorch/pytorch/issues/3749#issuecomment-374006211
# - https://pytorch.org/docs/stable/bottleneck.html

def param_count(model):
  return sum(p.numel() for p in model.parameters())
