import numpy as np
import torch


def pad(tensor, shape, value=0):
  if tensor.shape == shape: return tensor

  if isinstance(tensor, np.ndarray):
    padded = np.zeros(shape, dtype=tensor.dtype)
  else:
    padded = torch.zeros(shape)
  if value != 0:
    padded[:] = value

  padded[tuple(map(slice, tensor.shape))] = tensor
  return padded


def pad_to_largest(tensors, value=0):
  shape = tuple(max(dim) for dim in zip(*(t.shape for t in tensors)))
  return [pad(t, shape, value) for t in tensors]