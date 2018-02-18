

def resolve(x, lookups=None):
  pass


def get_var(x, grads=True, volatile=False):
  if isinstance(x, list):
    pass

  return x

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


def observe(func):
  """Decorator that allows you to observe inputs and outputs of a function"""
  pass