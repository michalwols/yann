from collections import OrderedDict

from .base import Callback


class Checkpoint(Callback):
  dist_placement = 0

  def __init__(self, freq=1, save_on_end=True):
    self.freq = freq
    self.paths = OrderedDict()

    self.save_on_end = save_on_end

  def on_epoch_end(self, epoch, loss=None, metrics=None, trainer=None):
    if epoch % self.freq == 0:
      self.paths[trainer.num_steps] = trainer.checkpoint()

  def on_train_end(self, trainer=None):
    if (
      self.save_on_end
      and trainer.num_steps > 10
      and trainer.num_steps not in self.paths
    ):
      self.paths[trainer.num_steps] = trainer.checkpoint()
