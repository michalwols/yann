from torch import nn


def residual(input, block, identity=None):
  p = block(input)
  if identity:
    input = identity(input)
  return input + p


class Residual(nn.Module):
  def __init__(self, *block, identity=None, activation=None):
    super().__init__()
    from . import Stack

    self.block = block[0] if len(block) == 1 else Stack(*block)
    self.identity = identity
    self.activation = activation

  def forward(self, input):
    residual = self.block(input)
    if self.identity:
      input = self.identity(input)

    input = input + residual
    if self.activation:
      return self.activation(input)
    else:
      return input
