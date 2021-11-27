from .base import Callback

import torch


class EarlyStopping(Callback):
  def __init__(self):
    pass


class StopOnNaN(Callback):
  def on_step_end(self, index, inputs, targets, outputs, loss, trainer=None):
    if torch.isnan(loss).any() \
       or torch.isinf(loss).any():
      print('NaN or Inf detected, stopping training')
      trainer.stop()