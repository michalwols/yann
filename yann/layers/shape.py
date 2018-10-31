from torch.nn import Module


class Reshape(Module):
  method = None

  def __init__(self, *dims):
    super(Reshape, self).__init__()
    self.dims = dims

  def forward(self, input):
    return getattr(input, self.method)(*self.dims)

  def state_dict(self, destination=None, prefix='', keep_vars=False):
    return {
      'dims': self.dims
    }

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
    return input.view(input.size(0), -1)


class FlattenSequences(Module):
  def forward(self, input, *rest):
    return flatten_sequence(input)


def flatten_sequence(seq_batch):
  seq_len, batch_size, *rest = seq_batch.size()
  return seq_batch.view(seq_len * batch_size, -1)
