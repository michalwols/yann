from torch import nn

from yann.modules import Stack
from yann.params import HyperParams


class DropPath(nn.Module):
  def __init__(self, drop_prob: float = 0.0):
    super().__init__()


class SqueezeExcitation(Stack):
  def __init__(self, in_features: int, out_features: float):
    super().__init__(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(),
      nn.BatchNorm2d(),
      nn.ReLU(),
      nn.Conv2d(),
      nn.Sigmoid(),
    )


class Params(HyperParams):
  pass


class SFCNN(nn.Module):
  Activation = nn.ReLU
