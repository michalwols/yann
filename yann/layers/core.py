
from torch.nn import Parameter, Module

from utils import lazy


class Params:
  def __init__(self, layer):
    self.layer = layer
    self.values = {}

  def __setitem__(self, key, value):
    self.values[key] = Parameter(value)

  def __getitem__(self, item):
   if item in self.values:
     return self.values[item]
   return getattr(self, item)

  def __len__(self):
    return

  def freeze(self):
    for p in self:
      p.requires_grad = False

  @property
  def initialized(self):
    return False

  def all(self):
    for p in self.values.values():
      yield p
    for l in self.layer.layers:
      yield from l.all()


class Grads:
  def __init__(self, layer):
    self.layer = layer

  def zero(self, deep=True):
    if deep:
      for g in self.all():
        g.data.zero_()

  def all(self):
    for p in self.layer.params.all():
      if p.grad:
        yield p.grad


class Layers:
  def __getitem__(self, item):
    """
    TODO:
    - allow indexing by class or class name to get all of that type
    """
    pass

  def breadth_first(self):
    pass

  def depth_first(self):
    pass


class Function:
  def __call__(self, *args, **kwargs):
    pass

  def serialize(self, **extra_args):
    pass

  def save(self):
    pass

  @classmethod
  def load(cls, *args, **kwargs):
    pass


class Partial(Function):
  def __init__(self, func, *args, **kwargs):
    self.func = func
    self.args = args
    self.kwargs = kwargs


class Layer(Function):
  def __init__(self, *args, **kwargs):
    self._is_training = True
    self.init = None
    self.name = None

  @lazy
  def layers(self):
    return Layers()

  @lazy
  def params(self):
    return Params(self)

  @lazy
  def grads(self):
    return Grads(self)

  def initialize(self, *args, **kwargs):
    pass

  def eval(self, *inputs, **kwargs):
    pass

  def update(self):
    pass

  def __call__(self, *inputs, **kwargs):
    return self.eval(*inputs, **kwargs)

  def infer_shape(self, *inputs, **kwargs):
    pass

  @property
  def is_training(self):
    return self._is_training

  @is_training.setter
  def is_training(self, val):
    self._is_training = val
    # TODO: propagate changes to sub layers

  def serialize(self, **extra_args):
    pass

  def save(self):
    pass

  def load(self, path):
    pass

  def device(self, name, num=None):
    pass

class Dense(Layer):
  pass


class Activation(Layer):
  pass


class Dropout(Layer):
  def __init__(self, p=.5):
    super(Dropout, self).__init__()
    self.p = p


class Softmax(Activation):
  def __init__(self, log=False):
    super(Softmax, self).__init__()
    self.log = log


class Reshape(Layer):
  pass

class Flatten(Reshape):
  pass

class Merge(Layer):
  pass


class Residual(Layers):
  def __init__(self, *towers):
    pass


class Stack(Layer):
  def __init__(self, *layers, name=None):
    super(Stack, self).__init__()
    self.layers = []
    self.add(layers)

    self.name = name

  def add(self, *layers):
    for l in layers:
      if l:
        # TODO: maybe wrap normal functions in Partial
        self.layers.append(l)