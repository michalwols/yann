from torch import nn


class DepthwiseConv2d(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias=True, padding_mode='zeros'):
    if out_channels % in_channels:
      raise ValueError('out_channels must be a multiple of in_channels')
    super(DepthwiseConv2d, self).__init__(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      dilation=dilation,
      bias=bias,
      padding_mode=padding_mode,
      groups=in_channels
    )


class DepthwiseSeparableConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias=True, padding_mode='zeros'):
    super(DepthwiseSeparableConv2d, self).__init__()

    self.depthwise = DepthwiseConv2d(in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      dilation=dilation,
      bias=bias,
      padding_mode=padding_mode)

    self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, input):
    x = self.depthwise(input)
    return self.pointwise(x)