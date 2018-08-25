from collections import defaultdict

class Callback:
  def on_train_start(self, trainer=None):
    pass

  def on_epoch_start(self, epoch, trainer=None):
    pass

  def on_batch_start(self, batch, inputs, targets, trainer=None):
    pass

  def on_batch_end(self, batch, inputs, targets, outputs, loss, trainer=None):
    pass

  def on_epoch_end(self, epoch, loss=None, metrics=None, trainer=None):
    pass

  def on_validation_start(self, trainer):
    pass

  def on_validation_batch(self, inputs, targets, outputs, trainer):
    pass

  def on_validation_end(self, targets=None, outputs=None, loss=None,
                        trainer=None):
    pass

  def on_train_end(self, trainer=None):
    pass

  def on_batch_error(self, batch, error, trainer=None):
    pass

  def on_error(self, error, trainer=None):
    pass


class FunctionCallback(Callback):
  _valid_names = {
    'train_start',
    'epoch_start',
    'batch_start',
    'batch_end',
    'epoch_end',
    'validation_start',
    'validation_batch',
    'validation_end',
    'train_end',
    'error',
  }

  def __init__(self, callbacks=None):
    self.callbacks = callbacks or defaultdict(list)

  def on(self, event, callback):
    if event not in self._valid_names:
      raise ValueError(
        f'{event} is not a valid option, '
        f'must be one of {self._valid_names}')
    callbacks = self.callbacks[event]
    if callback not in callbacks:
      callbacks.append(callback)

  def on_train_start(self, *args, **kwargs):
    for f in self.callbacks['train_start']:
      f(*args, **kwargs)

  def on_epoch_start(self, *args, **kwargs):
    for f in self.callbacks['epoch_start']:
      f(*args, **kwargs)

  def on_batch_start(self,*args, **kwargs):
    for f in self.callbacks['batch_start']:
      f(*args, **kwargs)

  def on_batch_end(self, *args, **kwargs):
    for f in self.callbacks['batch_end']:
      f(*args, **kwargs)

  def on_epoch_end(self, *args, **kwargs):
    for f in self.callbacks['epoch_end']:
      f(*args, **kwargs)

  def on_validation_start(self, *args, **kwargs):
    for f in self.callbacks['validation_start']:
      f(*args, **kwargs)

  def on_validation_batch(self, *args, **kwargs):
    for f in self.callbacks['validation_batch']:
      f(*args, **kwargs)

  def on_validation_end(self, *args, **kwargs):
    for f in self.callbacks['validation_end']:
      f(*args, **kwargs)

  def on_train_end(self, *args, **kwargs):
    for f in self.callbacks['train_end']:
      f(*args, **kwargs)

  def on_error(self, *args, **kwargs):
    for f in self.callbacks['error']:
      f(*args, **kwargs)