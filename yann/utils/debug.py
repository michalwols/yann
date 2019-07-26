import torch
import gc
import numpy as np
from torch import nn


def iter_allocated_tensors():
  for x in gc.get_objects():
    try:
      if torch.is_tensor(x) or (hasattr(x, 'data') and torch.is_tensor(x.data)):
        yield x
    except:
      pass


def bytecount(tensor: torch.Tensor):
  return tensor.numel() * tensor.element_size()


def receptive_field(module: nn.Module, input_shape=(1, 1, 256, 256)):
  module.zero_grad()
  input = torch.ones(*input_shape, requires_grad=True)
  output = module(input)
  indicator = torch.zeros_like(output)
  indicator[(0, 0, *(n // 2 for n in indicator.shape[2:]))] = 1

  loss = torch.mean(output * indicator)
  loss.backward()

  grad = input.grad.numpy()
  nonzero_indices = np.where(grad != 0)
  return tuple(x.max() - x.min() + 1 for x in nonzero_indices)