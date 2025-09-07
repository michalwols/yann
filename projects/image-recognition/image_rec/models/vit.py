from urllib.parse import ResultBase

import torch
import torch.nn.functional as F
from torch import nn

from yann.modules import Residual, Stack


class PatchEmbedding(nn.Module):
  def __init__(self, patch_size=16, in_channels=3, out_channels=768):
    super(PatchEmbedding, self).__init__()

    self.conv = nn.Conv2d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=patch_size,
      stride=patch_size,
      bias=False,
    )

  def forward(self, batch):
    """

    :param batch:  (N, C, H, W)
    :return:
    """

    return self.conv(batch).flatten(2).transpose(1, 2)  # (N, T, C)


class SwiGLU(nn.Module):
  def __init__(self, in_channels, out_channels=None):
    super(SwiGLU, self).__init__()

    if not out_channels:
      out_channels = in_channels

    self.linear = nn.Linear(in_channels, out_channels)
    self.linear2 = nn.Linear(in_channels, out_channels)

  def forward(self, x):
    weights = F.silu(self.linear(x))
    logits = self.linear2(x)
    return weights * logits


class TransformerBlock(nn.Module):
  def __init__(self):
    super().__init__()

    self.layers = Stack(
      Residual(RMSNorm(), nn.MultiheadAttention()),
      Residual(RMSNorm(), SwiGLU()),
    )


class ViT(nn.Module):
  def __init__(
    self,
  ):
    super(ViT, self).__init__()

    self.path_embed = PatchEmbedding(
      patch_size=16,
    )
