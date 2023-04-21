from collections import OrderedDict

import yann.callbacks
from yann.utils import camel_to_snake
from yann.callbacks.base import Callback


class Events:
  init = 'init'
  train_start = 'train_start'
  train_end = 'train_end'
  epoch_start = 'epoch_start'
  epoch_end = 'epoch_end'
  step_start = 'step_start'
  step_end = 'step_end'
  step_error = 'step_error'
  error = 'error'


def callback(method):
  def wrapped_method(self, *args, **kwargs):
    ret = method(self, *args, **kwargs)
    for callback_ in self:
      if hasattr(callback_, 'enabled') and not callback_.enabled:
        continue
      if hasattr(callback_, method.__name__):
        getattr(callback_, method.__name__)(*args, **kwargs)
    return ret

  return wrapped_method


class Callbacks(Callback):
  def __init__(self, *ordered, **named):
    if len(ordered) == 1 and not named:
      if isinstance(ordered[0], (list, tuple)):
        ordered = ordered[0]
      elif isinstance(ordered[0], dict):
        named = ordered[0]
      elif isinstance(ordered[0], Callbacks):
        named = ordered[0]._callbacks

    ordered = {self._get_name(x): x for x in ordered}
    ordered.update(named)
    self._callbacks = OrderedDict(ordered)

  def move_to_end(self, name, last=True):
    self._callbacks.move_to_end(name, last=last)

  def move_to_start(self, name):
    self.move_to_end(name, last=False)

  def __getitem__(self, key):
    return self._callbacks[key]

  def __getattr__(self, name):
    if name in self._callbacks:
      return self._callbacks[name]
    raise AttributeError()

  def __setattr__(self, key, value):
    if isinstance(value, Callback):
      self._callbacks[key] = value
    else:
      super(Callbacks, self).__setattr__(key, value)

  def __delattr__(self, name):
    del self._callbacks[name]

  def __contains__(self, name):
    return name in self._callbacks

  def __iter__(self):
    yield from self._callbacks.values()

  def __len__(self):
    return len(self._callbacks)

  def __bool__(self):
    return len(self._callbacks) > 0

  def __str__(self):
    return f"Callbacks({', '.join(f'{k}={v}' for k,v in self._callbacks.items())}"

  def append(self, callback):
    name = self._get_name(callback)
    self._callbacks[name] = callback

  def _get_name(self, x):
    return camel_to_snake(x.__class__.__name__)

  def on(self, event, callback=None):
    if 'function_callback' not in self._callbacks:
      self.function_callback = yann.callbacks.FunctionCallback()

    if callback:
      self.function_callback.on(event, callback)
      return self
    else:
      def decorated(func):
        self.function_callback.on(event, func)
        return func

      return decorated

  @callback
  def on_init(self, trainer=None):
    pass

  @callback
  def on_train_start(self, trainer=None):
    pass

  @callback
  def on_epoch_start(self, epoch: int, trainer=None):
    pass

  @callback
  def on_step_start(self, index: int, inputs, targets, trainer=None):
    pass

  @callback
  def on_step_end(self, index: int, inputs, targets, outputs, loss, trainer=None):
    pass

  @callback
  def on_step_error(self, index: int, error, trainer=None):
    pass

  @callback
  def on_error(self, error, trainer=None):
    pass

  @callback
  def on_epoch_end(self, epoch: int, trainer=None):
    pass

  @callback
  def on_validation_start(self, trainer=None):
    pass

  @callback
  def on_validation_batch(self, inputs, targets, outputs, trainer=None):
    pass

  @callback
  def on_validation_end(self, targets=None, outputs=None, loss=None, trainer=None):
    pass

  @callback
  def on_train_end(self, trainer=None):
    pass

