from torch.utils.tensorboard import SummaryWriter
from torch import nn

from .base import Callback


class Tensorboard(Callback):
  def __init__(self, root=None, trainer=None, writer=None):
    self.root = root
    self.trainer = trainer
    self.writer = writer

    self.logged_model_graph = False

  def on_train_start(self, trainer=None, **kwargs):
    if self.root is None:
      self.root = trainer.paths.tensorboard
    self.trainer = trainer

    if self.writer is None:
      self.writer = SummaryWriter(self.root)

    if self.trainer.params:
      self.writer.add_hparams(dict(self.trainer.params), {})

  def on_batch_start(self, inputs=None, **kwargs):
    if not self.logged_model_graph:
      self.writer.add_graph(self.trainer.model, inputs)
      self.logged_model_graph = True

  def log(self, val, tag=None, step=None, **kwargs):
    if step is None and self.trainer:
      step = self.trainer.num_steps

    if isinstance(val, nn.Module):
      self.writer.add_graph(val, kwargs.get('input'), verbose=kwargs.get('verbose', False))

  def show(self):
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('load_ext tensorboard')
    ipython.magic(f'tensorboard --logdir {self.root}')

  def close(self):
    if self.writer:
      self.writer.close()

  def flush(self):
    if self.writer:
      self.writer.flush()

