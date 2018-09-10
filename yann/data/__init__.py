from .loaders import TransformLoader
from .datasets import TransformDataset
from .classes import Classes


def get_dataset_name(x):
  while hasattr(x, 'dataset'):
    x = x.dataset

  if hasattr(x, 'name'):
    return x.name

  return x.__class__.__name__