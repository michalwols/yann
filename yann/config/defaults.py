import torch
from pathlib import Path


class default:
  root = Path('~/.yann/').expanduser()
  torch_root = Path('~/.torch').expanduser()
  train_root = './runs/'
  datasets_root = torch_root / 'datasets'

  device = None
  if torch.cuda.is_available():
    device = torch.device('cuda')
  elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')

  batch_size = 32
  num_workers = None
  optimizer = None

  callbacks = None

  checkpoint_name_format = ''

  ddp_find_unused_parameters = True

  @classmethod
  def dataset_root(cls, dataset):
    if hasattr(dataset, 'root'):
      return dataset.root
    return str(cls.datasets_root / dataset.__name__)