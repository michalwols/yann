import logging
import math
import typing
from itertools import zip_longest
from typing import Union

import numpy as np
import torch


class DatasetWrapper:
  def __init__(self, dataset):
    self.dataset = dataset

  def __getitem__(self, item):
    return self.dataset[item]

  def __len__(self):
    return len(self.dataset)

  def __getattr__(self, item):
    # proxy attribute lookups to the wrapped dataset so the
    # wrappers can transparently wrap datasets without breaking
    # code that expects a plain dataset
    return getattr(self.dataset, item)

  # NOTE: need get and set state to avoid issues when pickling for multiprocessing
  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, state):
    self.__dict__.update(state)


class IncludeIndex(DatasetWrapper):
  def __init__(self, dataset, last=False):
    super(IncludeIndex, self).__init__(dataset)
    self.last = last

  def __getitem__(self, idx):
    z = super().__getitem__(idx)

    if isinstance(z, (tuple, list)):
      return (*z, idx) if self.last else (idx, *z)

    return (z, idx) if self.last else (idx, z)


class Sliceable(DatasetWrapper):
  def __init__(self, dataset, columnar=True):
    super().__init__(dataset)
    self.columnar = columnar

  def __getitem__(self, x):
    if isinstance(x, slice):
      d = [self.dataset[n] for n in range(*x.indices(len(self.dataset)))]
      if self.columnar:
        return list(zip(*d))
      return d
    return self.dataset[x]


class Subset(DatasetWrapper):
  @typing.overload
  def __init__(
    self,
    dataset: typing.Mapping,
    indices: Union[np.ndarray, torch.Tensor],
  ): ...

  @typing.overload
  def __init__(self, dataset: typing.Mapping, end: Union[float, int]): ...

  @typing.overload
  def __init__(
    self,
    dataset: typing.Mapping,
    start: Union[float, int],
    end: Union[float, int],
  ): ...

  def __init__(self, dataset, *args):
    super(Subset, self).__init__(dataset)

    self.start = None
    self.end = None
    self.indices = None

    if len(args) == 1:
      if isinstance(args[0], int):
        self.start = 0
        self.end = (
          args[0] if not (0 < args[0] < 1) else math.floor(len(dataset) * args[0])
        )
      elif isinstance(args[0], Union[np.ndarray, torch.Tensor]):
        self.indices = args[0]
    elif len(args) == 2:
      if 0 < args[1] <= 1:
        self.start = math.floor(len(dataset) * args[0])
        self.end = math.floor(len(dataset) * args[1])
      else:
        self.start, self.end = args

  def __len__(self):
    if self.indices is not None:
      return self.indices
    else:
      return self.end - self.start

  def __getitem__(self, index):
    if index >= len(self):
      raise IndexError(f'Index out of bounds {index}')
    if self.indices is not None:
      return self.dataset[self.indices[index]]
    else:
      return self.dataset[self.start + index]


class Slice(DatasetWrapper):
  def __init__(self, dataset, start=0, end=None):
    super(Slice, self).__init__(dataset)

    self.start = start
    self.end = end if end is not None else len(dataset)

  def __getitem__(self, idx):
    if idx >= len(self):
      raise IndexError(f'Index out of bounds {idx}')
    return self.dataset[self.start + idx]

  def __len__(self):
    return self.end - self.start


# class Subset(DatasetWrapper):
#   pass


class IndexedView(DatasetWrapper):
  def __init__(self, dataset, indices):
    super(IndexedView, self).__init__(dataset)

    self.indices = indices

  def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]

  def __len__(self):
    return len(self.indices)


class LookupCache(DatasetWrapper):
  """
  Cache lookups

  Default to storing cache in memory so avoid caching large datasets or large inputs

  pass a preprocess function to transform the outputs before caching (useful
  for fine tuning on precomputed embeddings)

  # TODO: add joblib.Memory wrapper as a disk based cache option
  """

  def __init__(self, dataset, preprocess=None, cache=None):
    super().__init__(dataset)

    self.cache = cache or {}
    self.preprocess = preprocess

  def __getitem__(self, item):
    if item in self.cache:
      return self.cache[item]

    x = super().__getitem__(item)
    if self.preprocess:
      x = self.preprocess(x)

    self.cache[item] = x
    return x

  def set(self, key, value):
    self.cache[key] = value

  def update(self, keys, values):
    for k, v in zip(keys, values):
      self.set(k, v)


class TransformDataset(DatasetWrapper):
  def __init__(self, dataset, transform):
    super().__init__(dataset)
    self.transforms = transform if isinstance(transform, tuple) else (transform,)

  def __getitem__(self, idx):
    return tuple(
      t(x) if t else x for (x, t) in zip_longest(self.dataset[idx], self.transforms)
    )

  def __repr__(self):
    return (
      f'{self.__class__.__name__}('
      f'\nDataset: {repr(self.dataset)}'
      f'\nTransforms: {repr(self.transforms)}'
      f'\n)'
    )


# class Noisy(DatasetWrapper):
#   def __init__(self, dataset, p=0.1):
#     super().__init__(dataset)
#     self.p = p
#
#
# class Imbalanced(DatasetWrapper):
#   pass
#
#
# class SemiSupervised(DatasetWrapper):
#   """TODO: hide subset of labels"""
#   pass


class SwallowErrors(DatasetWrapper):
  def __init__(self, dataset, errors=None):
    super(SwallowErrors, self).__init__(dataset)
    self.errors = errors or Exception

  def __getitem__(self, item):
    try:
      return self.dataset[item]
    except KeyboardInterrupt as e:
      raise e
    except self.errors as e:
      logging.warning(e)
      return None


class VariableLength(DatasetWrapper):
  def __init__(self, dataset, max_size=1e20):
    super(VariableLength, self).__init__(dataset)
    self.max_size = max_size

  def __len__(self):
    return self.max_size

  def __getitem__(self, idx):
    return self.dataset[idx % len(self.dataset)]
