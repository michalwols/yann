

class Params:
  def __setitem__(self, key, value):
    pass

  def __getitem__(self, item):
    pass

  def __len__(self):
    return 1

class Layers:
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
  pass


class Layer(Function):
  def __init__(self, *args, **kwargs):
    self.was_initialized = False
    self._is_training = True
    self.params = Params()
    self.init = None

  def initialize(self):
    if self.init:
      self.init(self.params)

  def eval(self, *inputs, **kwargs):
    pass

  def update(self):
    pass

  def __call__(self, *inputs, **kwargs):
    return self.eval(*inputs, **kwargs)

  def infer_shape(self, *inputs, **kwargs):
    pass

  def output_shape(self):
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


class Conv(Layer):
  def __init__(
      self,
      d,
      num_filters,
      shape,
      stride=1,
      pad=0,
      dilate=0,
      groups=1
  ):
    super(Conv, self).__init__()


class Conv1D(Conv):
  def __init__(self, num_filters, shape):
    super(Conv1D, self).__init__(d=1, num_filters=num_filters, shape=shape)

class Conv2D(Conv):
  def __init__(self, num_filters, shape):
    super(Conv2D, self).__init__(d=2, num_filters=num_filters, shape=shape)


class Pool(Layer):
  pass

class GlobalPool(Pool):
  # TODO: use mean, max
  pass

class AdaptivePool(Pool):
  pass

class Dense(Layer):
  pass


class Activation(Layer):
  pass


class Dropout(Layer):
  pass


class Softmax(Activation):
  pass


class Reshape(Layer):
  pass


class Stack(Layer):
  def __init__(self, *layers):
    super(Stack, self).__init__()
    self.layers = []
    self.add(layers)

  def add(self, *layers):
    for l in layers:
      if l:
        # TODO: maybe wrap normal functions in Partial
        self.layers.append(l)