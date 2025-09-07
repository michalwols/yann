import inspect
from functools import wraps


def with_args(func, *args, **kwargs):
  sig = inspect.signature(func)
  bound = sig.bind(*args, **kwargs)
  bound.apply_defaults()

  return func(*args, **kwargs), bound.arguments


class Tracker:
  def __init__(self):
    self.values = {}
    self.enabled = True
    self.log = False

  def args(self, key=None):
    if not self.enabled:
      return lambda x: x

    def decorator(func):
      k = key or func.__name__

      @wraps(func)
      def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        if self.log:
          print(k, bound.arguments)

        self.values[k] = bound.arguments
        return func(*args, **kwargs)

      return wrapper

    return decorator

  def attributes(
    self,
    key=None,
    obj=None,
    types=(list, tuple, int, float, bool),
    cast=None,
    private=False,
  ):
    key = key or obj.__class__.__name__
    for k, v in vars(obj).items():
      pass


track = Tracker()


"""
# TODO
- autocast new version
- compile
- more type annotations

- webdataset support
- huggingface support 
- support dict samples
  

"""
