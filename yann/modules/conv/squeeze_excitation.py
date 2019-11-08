from torch import nn
from ..stack import Stack
from ..residual import Residual


class SqueezeExcitation(Residual):

  def __init__(self, channels: int, reduction: int):
    """
    Args:
      channels:
      reduction: reduction factor, paper claims 16 was the best value
    """
    inner_channels = channels // reduction
    super().__init__(
      Stack(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, inner_channels, kernel_size=1, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, channels, kernel_size=1, padding=0),
        nn.Sigmoid()
      )
    )


SE = SqueezeExcitation