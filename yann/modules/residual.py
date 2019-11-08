from torch import nn

# TODO: handle downsampling logic

def residual(input, block, downsample=None):
  p = block(input)
  if downsample:
    input = downsample(input)
  return input + p


class Residual(nn.Module):
  def __init__(self, block, downsample=None):
    super().__init__()
    self.block = block
    self.downsample = downsample

  def forward(self, input):
    processed = self.block(input)
    if self.downsample:
      input = self.downsample(input)
    return input + processed
