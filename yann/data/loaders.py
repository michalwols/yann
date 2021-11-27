from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from typing import Union, Iterable, Optional, Callable

from ..datasets import TransformDataset
import yann


class LoopedDataLoader(DataLoader):
  """
  Reuse the same iterator for multiple epochs to avoid startup penalty of
  initializing it each time

  (inspired by loader from `timm`)

  # might be fixed here https://github.com/pytorch/pytorch/pull/35795
  """
  def __init__(self, *args, **kwargs):
    super(DataLoader, self).__init__(*args, **kwargs)
    self.__initialized = False
    self.batch_sampler = LoopSampler(self.batch_sampler)
    self.__initialized = True
    self.iterator = super().__iter__()

  def __len__(self):
    return len(self.batch_sampler.sampler)

  def __iter__(self):
    for i in range(len(self)):
      yield next(self.iterator)


class LoopSampler:
  def __init__(self, sampler):
    self.sampler = sampler

  def __iter__(self):
    while True:
      yield from self.sampler


class TransformLoader(DataLoader):
  def __init__(self, dataset, transform, **kwargs):
    super(TransformLoader, self
         ).__init__(TransformDataset(dataset, transform), **kwargs)


def loader(
  data: Union[str, Iterable, Dataset, DataLoader],
  transform: Optional[Callable] = None,
  **kwargs
):
  """instantiate a loader from a dataset name, dataset or loader"""
  if isinstance(data, DataLoader):
    return data
  if isinstance(data, str):
    data = yann.resolve.dataset(data)
  if transform:
    return TransformLoader(data, transform=transform, **kwargs)
  else:
    return DataLoader(data, **kwargs)
