import os
from glob import iglob
from typing import Any, Iterable, Union

import torch
from torch.utils import data

from ..data.classes import Classes
from .wrappers import (
  DatasetWrapper,
  IncludeIndex,
  IndexedView,
  LookupCache,
  Subset,
  TransformDataset,
)


class Dataset(data.Dataset):
  def state_dict(self):
    return {'name': self.__class__.__name__}


class SupervisedDataset(Dataset):
  def __init__(self):
    if not hasattr(self, 'inputs'):
      self.inputs = None
    if not hasattr(self, 'targets'):
      self.targets = None

  def __getitem__(self, idx):
    return self.inputs[idx], self.targets[idx]

  def __len__(self):
    return len(self.inputs)


class ClassificationDataset(SupervisedDataset):
  def __init__(self, classes: Union[Classes, Iterable[Any]]):
    super(ClassificationDataset, self).__init__()
    self.classes = classes if isinstance(classes, Classes) else Classes(classes)


class GlobDataset(Dataset):
  def __init__(self, pattern='**/*.*', limit=None):
    paths = []
    for n, p in enumerate(iglob(pattern, recursive=True)):
      if os.path.getsize(p) < 4000:
        continue
      paths.append(p)
      if limit and n >= limit:
        break
    self.paths = paths

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, idx):
    return (self.paths[idx], 0)


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
    try:
      from sklearn.datasets import load_digits
    except ImportError:
      raise ImportError(
        "TinyDigits requires scikit-learn. Install it with: pip install scikit-learn"
      )

    digits = load_digits(num_classes)
    super().__init__(
      torch.from_numpy(digits.images).unsqueeze(1).float(),
      torch.Tensor(digits.target).long(),
    )


# from .voc import VOCMultilabel
# from .coco import CocoMultilabel
# from .imagenette import Imagenette, Imagewoof
