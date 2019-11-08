from torch import nn
from ..stack import Stack


class ConvBlock(Stack):

  class default:
    conv = nn.Conv2d
    norm = nn.BatchNorm2d
    activation = nn.ReLU

  def __init__(
    self,
    in_channels=None,
    out_channels=None,
    kernel_size=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=None,
    padding_mode='zeros',
    conv=True,
    norm=None,
    activation=True,
    order=('conv', 'norm', 'activation'),
    **extra
  ):
    if conv is True:
      if bias is None:
        # disable conv bias if norm layer is used
        bias = norm is False
      conv = self.default.conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        padding_mode=padding_mode
      )
    if norm is True:
      if conv is not None:
        norm = self.default.norm(
          num_features=conv.in_channels
          if order.index('conv') > order.index('norm') else conv.out_channels
        )
      else:
        norm = self.default.norm(in_channels)

    if activation is True:
      activation = self.default.activation() if isinstance(
        self.default.activation, type
      ) else self.default.activation

    d = dict(**extra, conv=conv, norm=norm, activation=activation)
    # NOTE: this depends on dict ordering
    super(ConvBlock, self).__init__(**{k: d[k] for k in order})




class ConvBlock1x1(ConvBlock):
  def __init__(self, **kwargs):
    if kwargs.get('kernel_size', 1) != 1:
      raise ValueError(f'kernel_size must be `1` if provided as an argument, got kernel_size={kwargs["kernel_size"]}')
    super(ConvBlock1x1, self).__init__(kernel_size=1, padding=0, **kwargs)


class ConvBlock3x3(ConvBlock):
  def __init__(self, **kwargs):
    if kwargs.get('kernel_size', 3) != 3:
      raise ValueError(f'kernel_size must be `3` if provided as an argument, got kernel_size={kwargs["kernel_size"]}')
    super(ConvBlock3x3, self).__init__(kernel_size=3, padding=1, **kwargs)