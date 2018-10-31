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
