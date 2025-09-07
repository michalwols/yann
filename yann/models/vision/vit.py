import torch
from torch import nn

import yann
from yann.modules import Residual, Stack


class TransformerEncoderBlock(Stack):
  def __init__(self):
    super().__init__(
      token_mix=Residual(nn.LayerNorm(), nn.MultiheadAttention()),
      channel_mix=Residual(nn.LayerNorm(), nn.Linear()),
    )


class ViT(nn.Module):
  def __init__(self):
    super().__init__()

    self.tokenize = Stack()
    self.positional_encode = Stack()
    self.encoder = TransformerEncoder()

    self.backbone = Stack(
      self.tokenize,
      self.positional_encode,
      self.encoder,
    )
    self.classifier = Stack()

  def forward(self, x):
    x = self.tokenize(x)
    x = self.positional_encode(x)
    x = self.backbone(x)
    x = self.classifier(x)
    return x

  def compute_loss(self, x, y):
    pass
