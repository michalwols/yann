class BaseTrainer:
  def __init__(self):
    self._stop = False

  def run(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    self.run(*args, **kwargs)

  def batches(self, *args, **kwargs):
    return []

  def step(self, *args, **kwargs):
    pass

  def stop(self):
    self._stop = True

  def checkpoint(self):
    pass

  @classmethod
  def from_checkpoint(cls, path):
    pass
