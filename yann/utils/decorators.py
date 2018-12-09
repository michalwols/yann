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
