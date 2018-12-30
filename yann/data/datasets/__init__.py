from .wrappers import LookupCache, DatasetWrapper, TransformDataset, IncludeIndex


import torch
from torch.utils import data

from ..classes import Classes


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


