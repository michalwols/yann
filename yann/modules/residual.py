from torch import nn

# TODO: handle downsampling logic

def residual(input, block, identity=None):
  p = block(input)
  if identity:
    input = identity(input)
  return input + p


class Residual(nn.Module):
  def __init__(self, block, identity=None, activation=None):
    super().__init__()
    self.block = block
    self.identity = identity
    self.activation = activation

  def forward(self, input):
    residual = self.block(input)
    if self.identity:
      input = self.identity(input)

    input += residual
    if self.activation:
      return self.activation(input)
    else:
      return input