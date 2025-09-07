import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len, device=None):
    super().__init__()

    self.d_model = d_model
    self.max_len = max_len

    self.embeddings = nn.Embedding(max_len, d_model, device=device)
    self.init()

  def init(self, std=0.02):
    self.embeddings.weight.data.normal_(0, std=std)

  def forward(self, x, indices=None):
    if indices is None:
      indices = torch.arange(x.size(0), device=x.device)

    embeddings = self.embeddings(indices)
    return x + embeddings


class SinusoidalPositionalEncoding(nn.Module):
  """
  PE(pos, 2i) = sin(pos/10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
  """

  def __init__(self, d_model, max_len, device=None):
    super().__init__()

    self.d_model = d_model
    self.max_len = max_len

    embeddings = torch.zeros(max_len, d_model, device=device)
    indices = torch.arange(
      0,
      max_len,
      device=device,
      dtype=torch.float,
    ).unsqueeze(1)

    div = torch.exp(
      torch.arange(0, d_model, 2, device=device).float()
      * (-math.log(10000.0) / d_model),
    )

    embeddings[:, 0::2] = torch.sin(indices * div)
    embeddings[:, 1::2] = torch.cos(indices * div)

    self.register_buffer('embeddings', embeddings)

  def forward(self, x, indices=None):
    if indices is None:
      embeddings = self.embeddings[:, : x.size(1)]
    else:
      embeddings = self.embeddings[:, indices]
    return x + embeddings


class RoPE(nn.Module):
  def __init__(self, d_model, max_len, device=None):
    super().__init__()
    self.d_model = d_model
