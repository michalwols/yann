from torch import nn

from .conv import (
  ConvBlock,
  ConvBlock1x1,
  ConvBlock3x3,
  DepthwiseConv2d,
  DepthwiseSeparableConv2d,
  EfficientChannelAttention,
  MixConv,
  SqueezeExcitation,
)
from .residual import Residual
from .shape import (
  Flatten,
  FlattenSequences,
  Infer,
  Permute,
  Reshape,
  Squeeze,
  Transpose,
  View,
)
from .stack import Stack


class Init:
  def __init__(self, cls, *args, **kwargs):
    self.cls = cls
    self.args = args
    self.kwargs = kwargs

  def __call__(self, *args, **kwargs):
    return self.cls(*self.args, *args, **{**self.kwargs, **kwargs})


class TrainEvalSwitch(nn.Module):
  def __init__(self, train=None, eval=None):
    super().__init__()
    self.train = train
    self.eval = eval

  def forward(self, *args):
    if self.training:
      return self.train(*args) if self.train else args
    else:
      return self.eval(*args) if self.eval else args
