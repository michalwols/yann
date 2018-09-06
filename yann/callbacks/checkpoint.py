from .base import Callback


class Checkpoint(Callback):
  def __init__(self, freq=1, load_latest=False):
    self.freq = freq

    self.load_latest = load_latest
  def on_train_start(self, trainer=None):
    if self.load_latest:
      # trainer.load_checkpoint()
      pass

  def on_epoch_end(self, epoch, loss=None, metrics=None, trainer=None):
    if epoch % self.freq == 0 :
      trainer.checkpoint()

