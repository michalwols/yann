from torch import nn
import torch

from .utils import get_same_padding


class MixConv(nn.Module):
  convs: nn.ModuleList
  """
  
  From:
    MixConv: Mixed Depthwise Convolutional Kernels
    https://arxiv.org/abs/1907.09595
  
  
  NOTE: with MKLDNN and padding == 3 and depthwise=False there's a runtime error that crashes the interpreter
  so kernel_size = 7 might lead to crashes on CPUs (seeing this on a Mac) 
  https://github.com/pytorch/pytorch/issues/20583
  """
  def __init__(
      self,
      in_channels,
      out_channels,
      kernel_size=None,
      depthwise=True
  ):
    super(MixConv, self).__init__()

    if isinstance(kernel_size, int):
      self.kernel_sizes = [kernel_size]
    elif isinstance(kernel_size, (tuple, list)):
      self.kernel_sizes = list(kernel_size)
    elif kernel_size is None:
      if not isinstance(in_channels, (list, tuple)):
        raise ValueError('kernel_size must be provided if in_channels is not an iterable')
      self.kernel_sizes = [3 + 2 * n for n in range(len(in_channels))]

    if isinstance(in_channels, (list, tuple)):
      self.input_channel_counts = in_channels
    else:
      self.input_channel_counts = self.split_groups(in_channels, len(self.kernel_sizes))
    if isinstance(out_channels, (list, tuple)):
      self.output_channel_counts = out_channels
    else:
      self.output_channel_counts = self.split_groups(out_channels, len(self.kernel_sizes))

    if len(self.input_channel_counts) != len(self.output_channel_counts):
      raise ValueError(
        f'in_channels and out_channels should have same number of groups,'
        f' but got {len(self.input_channel_counts)} and {len(self.output_channel_counts)}'
      )

    self.convs = nn.ModuleList([
      nn.Conv2d(
        in_channels=ic,
        out_channels=oc,
        kernel_size=ks,
        groups=min(ic, oc) if depthwise else 1,
        padding=get_same_padding(ks)
      )
      for ic, oc, ks
      in zip(self.input_channel_counts, self.output_channel_counts, self.kernel_sizes)
    ])

  def __repr__(self):
    return f"MixConv({self.input_channel_counts}, {self.output_channel_counts}, kernel_sizes={self.kernel_sizes}, convs={self.convs})"

  def split_groups(self, num_channels, num_groups):
    channel_counts = [num_channels // num_groups] * num_groups
    channel_counts[0] += num_channels - sum(channel_counts)
    return channel_counts

  def forward(self, input):
    if len(self.convs) == 1:
      return self.convs[0](input)
    parts = torch.split(input, self.input_channel_counts, 1)
    outputs = [conv(part) for conv, part in zip(self.convs, parts)]
    return torch.cat(outputs, 1)


