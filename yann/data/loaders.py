from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from typing import Union, Iterable, Optional, Callable

from ..datasets import TransformDataset
import yann


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
  if isinstance(data, (Dataset, Iterable)):
    if transform:
      return TransformLoader(data, transform=transform, **kwargs)
    else:
      return DataLoader(data, **kwargs)
