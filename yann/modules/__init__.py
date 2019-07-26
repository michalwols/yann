from .shape import View, Flatten, Infer
from .stack import Stack


class Init:
  def __init__(self, cls, *args, **kwargs):
    self.cls = cls
    self.args = args
    self.kwargs = kwargs

  def __call__(self, *args, **kwargs):
    return self.cls(*self.args, *args, **{**self.kwargs, **kwargs})