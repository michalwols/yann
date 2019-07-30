import logging

import torch

from .registry import Registry, pass_args

## Configure Registry

registry = Registry()

# Datasets
import torchvision.datasets

from torch.utils.data import Dataset

registry.dataset.index(
  torchvision.datasets,
  types=(Dataset,),
  init=lambda D, root=None, download=True, **kwargs: \
    D(root=root or f'~/.torch/datasets/{D.__name__}',
      download=download)
)

from ..data.datasets import imagenette
registry.dataset.index(
  imagenette,
  types=(Dataset,),
  init=lambda D, root=None, download=True, **kwargs: \
    D(root=root or f'~/.torch/datasets/',
      download=download)
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

from ..modules import loss

registry.loss.update((
  F.cross_entropy,
  F.binary_cross_entropy,
  F.binary_cross_entropy_with_logits,
  loss.soft_target_cross_entropy,
  loss.SoftTargetCrossEntropyLoss
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


def is_public_callable(x):
  return (
      hasattr(x, '__name__')
      and not x.__name__.startswith('_')
      and callable(x)
  )


registry.model.torchvision.index(
  models,
  init=pass_args,
  include=is_public_callable
)

try:
  import pretrainedmodels.models
except ImportError:
  logging.debug("Couldn't register pretrainedmodels models because it's not "
                "installed")
else:
  registry.model.pretrainedmodels.index(
    pretrainedmodels.models,
    init=pass_args,
    include=is_public_callable
  )

from .. import metrics
registry.metric.update((
  metrics.accuracy,
  metrics.average_precision,
  metrics.average_precision_at_k,
  metrics.coverage_error,
))

registry.metric.register(
  metrics.top_3_accuracy,
  name='top_3_accuracy'
)

registry.metric.register(
  metrics.top_5_accuracy,
  name='top_5_accuracy'
)

registry.metric.register(
  metrics.top_10_accuracy,
  name='top_10_accuracy'
)