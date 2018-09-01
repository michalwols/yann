from itertools import zip_longest
from torch.utils.data import Dataset
from .classes import Classes


class TransformDataset(Dataset):
  def __init__(self, dataset, transforms):
    super().__init__()
    self.dataset = dataset
    self.transforms = transforms if isinstance(transforms, tuple) else (
      transforms,)

  def __len__(self):
    return len(self.dataset)

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


class Noisy(Dataset):
  def __init__(self, dataset, p=0.1):
    self.clean_dataset = dataset
    self.p = p

  def __len__(self):
    return len(self.clean_dataset)

  def __getitem__(self, idx):
    x, y = self.clean_dataset[idx]



class Imbalanced(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset



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