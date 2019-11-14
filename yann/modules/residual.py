from torch import nn

# TODO: handle downsampling logic

def residual(input, block, identity=None):
  p = block(input)
  if identity:
    input = identity(input)
  return input + p


class Residual(nn.Module):
  def __init__(self, block, identity=None):
    super().__init__()
    self.block = block
    self.identity = identity

  def forward(self, input):
    processed = self.block(input)
    if self.identity:
      input = self.identity(input)
    return input + processed
