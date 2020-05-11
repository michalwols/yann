from functools import wraps

class lazy:
  __slots__ = 'method', 'name'

  def __init__(self, method):
    self.method = method
    self.name = method.__name__

  def __get__(self, obj, cls):
    if obj:
      val = self.method(obj)
      setattr(obj, self.name, val)
      return val


class RobustFunction:
  def __init__(self, func, exceptions=Exception, default=None):
    self.func = func
    self.exceptions = exceptions
    self.default = default

  def __call__(self, *args, **kwargs):
    if not self.exceptions:
      return self.func(*args, **kwargs)
    try:
      r = self.func(*args, **kwargs)
    except self.exceptions:
      r = self.default
    return r


def robust(x=None, exceptions=Exception, default=None):
  if callable(x):
    return RobustFunction(func=x, exceptions=exceptions, default=default)
  else:
    def decorator(x):
      return RobustFunction(func=x, exceptions=exceptions, default=default)

    return decorator


class FunctionTracker:
  def __init__(self, func, sanitize=None):
    self.func = func
    self.history = []
    self.sanitize = sanitize

  def __call__(self, *args, **kwargs):
    r = self.func(*args, **kwargs)
    if self.sanitize:
      r = self.sanitize(r)
    self.history.append(r)
    return r


def track(x=None, sanitize=None):
  if callable(x):
    return FunctionTracker(func=x)
  else:
    def decorator(x):
      return FunctionTracker(func=x, sanitize=sanitize)

    return decorator