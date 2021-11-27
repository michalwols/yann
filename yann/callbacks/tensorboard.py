from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from .base import Callback


log = logging.getLogger(__name__)

class Tensorboard(Callback):
  writer: SummaryWriter
  def __init__(self, root=None, trainer=None, writer=None):
    self.root = root
    self.trainer = trainer
    self.writer = writer

    self.logged_model_graph = False

  def _init_writer(self):
    self.writer = SummaryWriter(self.root)

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
      # static method (Tensorboard.show())
      ipython.magic(f'tensorboard --logdir {self}')
    else:
      ipython.magic(f'tensorboard --logdir {root or self.root or "./"}')

  def close(self):
    if self.writer:
      self.writer.close()

  def flush(self):
    if self.writer:
      self.writer.flush()

  def on_train_start(self, trainer=None, **kwargs):
    if self.root is None:
      self.root = trainer.paths.tensorboard
    self.trainer = trainer

    if self.writer is None:
      self._init_writer()

  def on_train_end(self, trainer=None):
    if self.trainer and self.trainer.params and self.trainer.history.val_metrics:
      try:
        self.writer.add_hparams(
          dict(self.trainer.params),
          self.trainer.history.val_metrics.summary()
        )
      except ValueError as e:
        log.error(e)

  def on_step_start(self, inputs=None, **kwargs):
    if not self.logged_model_graph:
      self.writer.add_graph(self.sanitize_model(self.trainer.model), inputs)
      self.logged_model_graph = True

  def on_step_end(
    self,
    index=None,
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

  def sanitize_model(self, model):
    if isinstance(model, nn.DataParallel):
      return model.module
    return model