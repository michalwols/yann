import types

import torch

from . import place
from .classes import Classes
from .loaders import TransformLoader
from .transform import Transformer


def get_name(x):
  if hasattr(x, 'name'):
    return x.name

  return x.__class__.__name__


def get_dataset_name(x):
  while hasattr(x, 'dataset'):
    x = x.dataset

  if hasattr(x, 'name'):
    return x.name

  return x.__class__.__name__


def batches(*tensors, size=32, shuffle=False, order=None):
  if len(tensors) == 1 and isinstance(tensors[0], str):
    # assume a registered dataset name was passed (like batches('MNIST'))
    import yann

    tensors = (yann.resolve.dataset(tensors[0]),)
  if shuffle:
    order = torch.randperm(len(tensors[0]))

  if len(tensors) == 1:
    for i in range(0, len(tensors[0]), size):
      if order is not None:
        indices = order[i : i + size]
        yield tensors[0][indices]
      else:
        yield tensors[0][i : i + size]
  else:
    for i in range(0, len(tensors[0]), size):
      if order is not None:
        indices = order[i : i + size]
        yield tuple(t[indices] for t in tensors)
      else:
        yield tuple(t[i : i + size] for t in tensors)


def unbatch(batches):
  return (x for b in batches for x in b)


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
      yield sequence[i : i + size]


def loop(items):
  while True:
    yield from items


def shuffle(*sequences):
  order = torch.randperm(len(sequences[0]))
  return (
    [s[i] for i in order] if isinstance(s, (tuple, list)) else s[order]
    for s in sequences
  )


def flatten(x, out=None, prefix='', sep='.'):
  """
  Flatten nested dict
  """
  out = out if out is not None else {}

  if isinstance(x, dict):
    for k in x:
      flatten(
        x[k],
        out=out,
        prefix=f'{prefix}{sep if prefix else ""}{k}',
        sep=sep,
      )
  elif isinstance(x, (list, tuple)):
    for k, v in enumerate(x):
      flatten(k, out=out, prefix=f'{prefix}{sep if prefix else ""}{k}', sep=sep)
  else:
    out[prefix] = x

  return out


def print_tree(root, indent=4):
  from pathlib import Path

  root = Path(root)
  print(f'{root}')
  for path in sorted(root.rglob('*')):
    depth = len(path.relative_to(root).parts)
    print(f'{" " * (depth * indent)} {path.name}')
