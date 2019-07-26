from yann.callbacks.base import Callback
from torch.utils.tensorboard import SummaryWriter


class Tensorboard(Callback):
  def __init__(self, root=None):
    self.root = root
    self.writer = None
    self.trainer = None

  def open(self):
    self.writer = SummaryWriter(self.root)

  def close(self):
    self.writer.close()

  def on_train_start(self, trainer=None):
    self.trainer = trainer
    if not self.root:
      self.root = self.trainer.root / 'tensorboard'
    self.open()

  def on_train_end(self, trainer=None):
    self.close()

  def help(self):
    print(
      f'''
      # Viewing in Jupyter:
      %load_ext tensorboad
      %tensorboard --logdir={self.root}
      '''
    ) 