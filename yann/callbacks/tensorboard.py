from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torch import nn

from .base import Callback


class Tensorboard(Callback):
  writer: SummaryWriter
  def __init__(self, root=None, trainer=None, writer=None):
    self.root = root
    self.trainer = trainer
    self.writer = writer

    self.logged_model_graph = False

  def _init_writer(self):
    self.writer = SummaryWriter(self.root)

  def on_train_start(self, trainer=None, **kwargs):
    if self.root is None:
      self.root = trainer.paths.tensorboard
    self.trainer = trainer

    if self.writer is None:
      self._init_writer()

    if self.trainer and self.trainer.params:
      self.writer.add_hparams(dict(self.trainer.params), {})

  def on_batch_start(self, inputs=None, **kwargs):
    if not self.logged_model_graph:
      self.writer.add_graph(self.trainer.model, inputs)
      self.logged_model_graph = True

  def on_batch_end(
    self,
    batch=None,
    inputs=None,
    targets=None,
    outputs=None,
    loss=None,
    trainer=None
  ):
    self.writer.add_scalar('train/loss', loss, global_step=trainer.num_steps)
    for metric, values in trainer.history.metrics.items():
      self.writer.add_scalar(f'train/{metric}', values[-1], global_step=len(values) - 1)

  def on_validation_end(self, targets=None, outputs=None, loss=None, trainer=None):
    for metric, values in trainer.history.val_metrics.items():
      self.writer.add_scalar(f'validation/{metric}', values[-1], global_step=len(values) - 1)

  def log(self, val, tag=None, step=None, **kwargs):
    if step is None and self.trainer:
      step = self.trainer.num_steps

    if isinstance(val, nn.Module):
      self.writer.add_graph(
        val,
        kwargs.get('input'),
        verbose=kwargs.get('verbose', False)
      )

  def show(self, root=None):
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext tensorboard')

    if isinstance(self, (str, Path)):
      # static method
      ipython.magic(f'tensorboard --logdir {self}')
    else:
      ipython.magic(f'tensorboard --logdir {root or self.root}')

  def close(self):
    if self.writer:
      self.writer.close()

  def flush(self):
    if self.writer:
      self.writer.flush()
