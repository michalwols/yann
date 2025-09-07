import gc

import numpy as np
import torch
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

  loss = torch.sum(output * indicator)
  loss.backward()

  grad = input.grad.numpy()
  nonzero_indices = np.where(grad != 0)
  return tuple(x.max() - x.min() + 1 for x in nonzero_indices)


def fill_dims(t, f=1):
  if t.ndim == 0:
    return
  for n, x in enumerate(t):
    t[n, ...] += n * f
    fill_dims(t[n], f / 10)

  return t
