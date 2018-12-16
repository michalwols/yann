from collections import Counter
from inspect import isclass

from torch import nn

from ..utils import camel_to_snake


class Stack(nn.Module):
  """

  NOTE: this depends on dict ordering, which is only guaranteed for python >=3.6
  """

  def __init__(self, *layers, **named_layers):
    super().__init__()

    layer_counts = Counter()
    for l in layers:
      if l:
        name = camel_to_snake(l.__class__.__name__)
        num = layer_counts[name]
        self.add_module(name + str(num), l)
        layer_counts[name] += 1
    for name, l in named_layers.items():
      if l:
        self.add_module(name, l)

  def forward(self, x):
    for l in self.children():
      x = l(x)
    return x

  def trace(self, x):
    for l in self.children():
      x = l(x)
      yield l, x

  def __len__(self):
    return len(self._modules)

  def __getitem__(self, x):
    if isinstance(x, slice):
      layers = list(self.children())

      start, stop = x.start, x.stop
      if isinstance(x.start, nn.Module):
        start = layers.index(x.start)
      if isinstance(x.stop, nn.Module):
        stop = layers.index(x.stop)

      return self.__class__(
        **dict(list(self.named_children())[start:stop:x.step])
      )
    elif isclass(x):
      return [m for m in self.children() if isinstance(m, x)]
    else:
      return list(self.children())[x]

  def __setitem__(self, x, value):
    key = list(self._modules.keys())[x]
    setattr(self, key, value)
