import torch

from .utils.registry import Registry

## Configure Registry

registry = Registry()

# Datasets
from torchvision import datasets

from torch.utils.data import Dataset

registry.dataset.index(
  datasets,
  types=(Dataset,),
  init=lambda D: D(root=f'~/.torch/datasets/{D.__name__}', download=False)
)

# Losses
registry.loss.index(
  torch.nn.modules.loss,
  types=(torch.nn.modules.loss._Loss,),
  get_names=lambda x:
  (x.__name__, x.__name__[:-len('Loss')]) if x.__name__.endswith('Loss') else (
    x.__name__,)
)

import torch.nn.functional as F

registry.loss.index(
  F,
  include=lambda x: hasattr(x, '__name__') and 'loss' in x.__name__.lower(),
  get_names=lambda x: (x.__name__, x.__name__[:-len('_loss')]) if
  x.__name__.endswith('_loss') else (x.__name__,)
)

registry.loss.update((
  F.cross_entropy,
  F.binary_cross_entropy,
  F.binary_cross_entropy_with_logits
))

# Optimizers
from torch.optim import optimizer

registry.optimizer.index(
  torch.optim,
  types=(optimizer.Optimizer,)
)

registry.optimizer['SGD'].init = \
  lambda SGD, params, lr=.01, momentum=.9, weight_decay=1e-4: \
    SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

# LR Schedulers
from torch.optim import lr_scheduler

registry.lr_scheduler.index(
  lr_scheduler,
  types=(lr_scheduler._LRScheduler,)
)

# Models
from torchvision import models

registry.model.index(
  models,
  include=lambda x: hasattr(x, '__name__')
                    and (isinstance(x, torch.nn.Module)
                         or callable(x))
)
