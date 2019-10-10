import torch
import types

from .classes import Classes
from .datasets import TransformDataset
from .loaders import TransformLoader
from .transform import Transformer


def get_dataset_name(x):
  while hasattr(x, 'dataset'):
    x = x.dataset

  if hasattr(x, 'name'):
    return x.name

  return x.__class__.__name__


def batches(*tensors, size=32, shuffle=False, order=None):
  if shuffle:
    order = torch.randperm(len(tensors[0]))

  if len(tensors) == 1:
    for i in range(0, len(tensors[0]), size):
      if order is not None:
        indices = order[i:i+size]
        yield tensors[0][indices]
      else:
        yield tensors[0][i:i+size]
  else:
    for i in range(0, len(tensors[0]), size):
      if order is not None:
        indices = order[i:i+size]
        yield tuple(t[indices] for t in tensors)
      else:
        yield tuple(t[i:i+size] for t in tensors)


def chunk(sequence, size=32):
  if isinstance(sequence, types.GeneratorType):
    batch = []
    for x in sequence:
      batch.append(x)
      if len(batch) == size:
        yield batch
        batch = []
  else:
    for i in range(0, len(sequence), size):
      yield sequence[i:i+size]



def shuffle(*sequences):
  order = torch.randperm(len(sequences[0]))
  return (
     [s[i] for i in order] if isinstance(s, (tuple, list)) else s[order]
     for s in sequences
  )

def flatten(x, out=None, prefix='', sep='_'):
  """
  Flatten nested dict
  """
  out = out if out is not None else {}

  if isinstance(x, dict):
    for k in x:
      flatten(x[k], out=out, prefix=f"{prefix}{sep if prefix else ''}{k}", sep=sep)
  elif isinstance(x, (list, tuple)):
    for k, v in enumerate(x):
      flatten(k, out=out, prefix=f"{prefix}{sep if prefix else ''}{k}", sep=sep)
  else:
    out[prefix] = x

  return out