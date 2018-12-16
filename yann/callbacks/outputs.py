from .base import Callback


class OutputWriter(Callback):
  def __init__(self, freq):
    super().__init__()

  def write(self, *args, **kwargs):
    pass
