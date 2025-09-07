from typing import Dict

import torch

import yann
from yann.train import Trainer

from . import Callback


class GradualUnfreezing(Callback):
  trainer: Trainer

  def __init__(
    self,
    modules: Dict[int, torch.nn.Module],
    unfreeze=yann.unfreeze,
  ):
    self.modules = modules
    self.unfreeze = unfreeze

  def on_train_start(self, trainer=None):
    self.trainer = trainer

  def on_epoch_start(self, epoch=None, loss=None, metrics=None, trainer=None):
    modules = self.modules.get(epoch)
    if modules:
      self.make_trainable(modules)

  def make_trainable(self, modules):
    parameters = self.unfreeze(modules)

    # clone param group variables to avoid missing keys (used by things like lr_schedulers)
    param_group = {
      k: v for (k, v) in self.trainer.optimizer.param_groups[0].items() if k != 'params'
    }
    param_group['params'] = parameters

    self.trainer.optimizer.add_param_group(param_group)
