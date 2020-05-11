from torch import nn


class EfficientChannelAttention(nn.Module):
  """
  https://github.com/BangguWu/ECANet
  """
  def __init__(self, kernel_size=3):
    super().__init__()
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Conv1d(
      1, 1,
      kernel_size=kernel_size,
      padding=(kernel_size - 1) // 2,
      bias=False
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):
    x = self.pool(input)
    x = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
    x = self.sigmoid(x)

    return input * x.expand_as(input)

ECA = EfficientChannelAttention
