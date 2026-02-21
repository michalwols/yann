from torch.nn import Module

from ..exceptions import ShapeInferenceError


class Reshape(Module):
  method = None

  def __init__(self, *dims):
    super(Reshape, self).__init__()
    self.dims = dims

  def forward(self, input):
    return getattr(input, self.method)(*self.dims)

  def state_dict(self, destination=None, prefix='', keep_vars=False):
    return {'dims': self.dims}

  def load_state_dict(self, state_dict, strict=True):
    self.dims = state_dict['dims']


class Squeeze(Reshape):
  method = 'squeeze'


class Permute(Reshape):
  method = 'permute'


class Transpose(Reshape):
  method = 'transpose'


class View(Reshape):
  method = 'view'


class Flatten(Reshape):
  def forward(self, input, *rest):
    return input.reshape(input.size(0), -1)


def flatten(input):
  return input.reshape(input.size(0), -1)


class FlattenSequences(Module):
  def forward(self, input, *rest):
    return flatten_sequence(input)


def flatten_sequence(seq_batch):
  seq_len, batch_size, *rest = seq_batch.size()
  return seq_batch.view(seq_len * batch_size, -1)


class Infer(Module):
  def __init__(self, cls, *args, **kwargs):
    super(Infer, self).__init__()
    self.shape_dim = kwargs.pop('shape_dim', 1)
    self.cls = cls
    self.args = args
    self.kwargs = kwargs

    self.module = None

  @property
  def was_inferred(self):
    return self.module is not None

  def _infer_and_forward(self, x):
    try:
      self.module = self.cls(
        x.shape[self.shape_dim],
        *self.args,
        **self.kwargs,
      )
    except IndexError as e:
      raise ShapeInferenceError(
        f'Improper shape dim ({self.shape_dim}) selected for {self.cls} with input of shape {x.shape}',
      )
    # After inference, replace forward to skip the check on subsequent calls
    self.forward = self.module.forward
    return self.module(x)

  def forward(self, x):
    if self.module is None:
      return self._infer_and_forward(x)
    return self.module(x)

  @classmethod
  def shed(cls, module):
    """Replace all Infer nodes in a module with their initialized inner modules."""
    for name, child in list(module.named_children()):
      if isinstance(child, cls):
        if child.module is not None:
          setattr(module, name, child.module)
        else:
          raise RuntimeError(
            f'Cannot shed Infer node {name!r}: module has not been inferred yet. '
            'Run a forward pass first.',
          )
      else:
        cls.shed(child)
