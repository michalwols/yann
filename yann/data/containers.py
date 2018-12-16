from collections import OrderedDict
from collections import abc


class Container(abc.MutableMapping):
  def __init__(self, *args, **kwargs):
    items = OrderedDict(
      ('_arg' + str(n), v) for n, v in enumerate(args)
    )
    items.update(kwargs)

    self.__dict__.update(items)

    self._keys = list(items.keys())

    self.args = args
    self.kwargs = kwargs

  def __iter__(self):
    return (getattr(self, k) for k in self._keys)

  def __len__(self):
    return len(self._keys)

  def __getitem__(self, x):
    if isinstance(x, int):
      k = self._keys[x]
      return getattr(self, k)

    if isinstance(x, slice):
      return [getattr(self, k) for k in self._keys[x]]

  def __setitem__(self, key, value):
    if isinstance(key, int):
      setattr(self, self._keys[key], value)

  def __delitem__(self, key):
    pass


class Inputs(Container):
  pass


class Targets(Container):
  pass


class Outputs(Container):
  pass


class Samples:
  def __init__(self, inputs, targets):
    self.inputs = inputs
    self.targets = targets

  def __iter__(self):
    return (*self.inputs, *self.targets)
