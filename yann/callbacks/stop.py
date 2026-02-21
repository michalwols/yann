import torch

from .base import Callback


class EarlyStopping(Callback):
  """Stop training when a monitored metric has stopped improving.

  Args:
    patience: Number of epochs with no improvement after which training stops.
    min_delta: Minimum change to qualify as an improvement.
    mode: One of 'min' or 'max'. In 'min' mode, training stops when the
      metric has stopped decreasing; in 'max' mode it stops when the
      metric has stopped increasing.
    metric: Name of the metric to monitor. Defaults to 'loss'.
    verbose: Whether to print when stopping.
  """

  def __init__(
    self,
    patience=5,
    min_delta=0.0,
    mode='min',
    metric='loss',
    verbose=True,
  ):
    self.patience = patience
    self.min_delta = min_delta
    self.mode = mode
    self.metric = metric
    self.verbose = verbose

    self.best = None
    self.num_bad_epochs = 0

    if mode == 'min':
      self.is_better = lambda current, best: current < best - self.min_delta
    elif mode == 'max':
      self.is_better = lambda current, best: current > best + self.min_delta
    else:
      raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

  def on_epoch_end(self, epoch=None, trainer=None):
    val_metrics = trainer.history.val_metrics
    if self.metric not in val_metrics or len(val_metrics[self.metric]) == 0:
      return

    current = val_metrics[self.metric][-1]

    if self.best is None or self.is_better(current, self.best):
      self.best = current
      self.num_bad_epochs = 0
    else:
      self.num_bad_epochs += 1

    if self.num_bad_epochs >= self.patience:
      if self.verbose:
        print(
          f'EarlyStopping: {self.metric} has not improved for '
          f'{self.patience} epochs. Stopping training.',
        )
      trainer.stop()


class StopOnNaN(Callback):
  def on_step_end(self, index, inputs, targets, outputs, loss, trainer=None):
    if torch.isnan(loss).any() or torch.isinf(loss).any():
      print('NaN or Inf detected, stopping training')
      trainer.stop()
