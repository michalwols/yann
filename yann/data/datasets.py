import torch
from itertools import zip_longest
from torch.utils import data

from .classes import Classes


class Dataset(data.Dataset):
  def state_dict(self):
    return {
      'name': self.__class__.__name__
    }


class ClassificationDataset(Dataset):
  def __init__(self):
    pass


class InputsTargetsDataset(Dataset):
  def __init__(self, inputs, targets, transform=None):
    self.inputs = inputs
    self.targets = targets
    self.transform = transform
    self.classes = Classes(sorted(set(x for t in self.targets for x in t)))

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, idx):
    x, y = self.inputs[idx], self.targets[idx]
    if self.transform:
      x = self.transform(x)
    return x, y


class TinyDigits(data.TensorDataset):
  """
  Dataset of 8x8 digits, best used for testing
  """

  def __init__(self, num_classes=10):
    from sklearn.datasets import load_digits
    digits = load_digits(num_classes)
    super().__init__(
      torch.from_numpy(digits.images).unsqueeze(1).float(),
      torch.Tensor(digits.target).long()
    )


class DatasetWrapper:
  def __init__(self, dataset):
    self.dataset = dataset

  def __getitem__(self, item):
    return self.dataset[item]

  def __len__(self):
    return len(self.dataset)


class IncludeIndex(DatasetWrapper):
  def __getitem__(self, idx):
    z = super().__getitem__(idx)

    if isinstance(z, (tuple, list)):
      return (*z, idx)

    return (z, idx)


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
  def __init__(self, dataset, transforms):
    super().__init__(dataset)
    self.transforms = transforms if isinstance(transforms, tuple) else (
      transforms,)

  def __getitem__(self, idx):
    return tuple(
      t(x) if t else x
      for (x, t) in zip_longest(self.dataset[idx], self.transforms)
    )

  def __repr__(self):
    return (
      f'{self.__class__.__name__}('
      f'\nDataset: {repr(self.dataset)}'
      f'\nTransforms: {repr(self.transforms)}'
      f'\n)')


class Noisy(DatasetWrapper):
  def __init__(self, dataset, p=0.1):
    super().__init__(dataset)
    self.p = p


class Imbalanced(DatasetWrapper):
  pass
