def callback(method):
  def wrapped_method(self, *args, **kwargs):
    ret = method(self, *args, **kwargs)
    for c in self.callbacks:
      if hasattr(c, method.__name__):
        getattr(c, method.__name__)(*args, **kwargs, trainer=self)
    return ret

  return wrapped_method


class BaseTrainer:
  def __init__(self):
    self._stop = False

  def run(self, *args, **kwargs):
    pass

  def batches(self, *args, **kwargs):
    return []

  def step(self, *args, **kwargs):
    pass

  def __call__(self, *args, **kwargs):
    self.run(*args, **kwargs)

  def stop(self):
    self._stop = True

  def checkpoint(self):
    pass

  @classmethod
  def from_checkpoint(cls, path):
    pass

  @callback
  def on_train_start(self):
    pass

  @callback
  def on_epoch_start(self, epoch):
    pass

  @callback
  def on_batch_start(self, batch, inputs, targets):
    pass

  @callback
  def on_batch_end(self, batch, inputs, targets, outputs, loss):
    pass

  @callback
  def on_batch_error(self, batch, error):
    pass

  @callback
  def on_error(self, error):
    pass

  @callback
  def on_epoch_end(self, epoch):
    pass

  @callback
  def on_validation_start(self):
    pass

  @callback
  def on_validation_batch(self, inputs, targets, outputs):
    pass

  @callback
  def on_validation_end(self, targets=None, outputs=None, loss=None):
    pass

  @callback
  def on_train_end(self):
    pass








