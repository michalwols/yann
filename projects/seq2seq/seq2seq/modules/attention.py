from typing import Callable

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class SelfAttention(nn.Module):
  def __init__(
    self,
    embed_dim: int,
    num_heads: int = 1,
    num_groups: int = 1,
    bias=False,
    dropout=None,
    mask=None,
    q_norm=None,
    k_norm=None,
    kv_cahce=None,
    kernel='sdpa',
  ):
    super().__init__()

    self.embed_dim = embed_dim
    self.num_heads = num_heads

    self.num_groups = num_groups

    self.head_dim = embed_dim // num_heads
    self.group_dim = embed_dim // num_groups

    self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.Wk = nn.Linear(embed_dim, self.group_dim, bias=bias)
    self.Wv = nn.Linear(embed_dim, self.group_dim, bias=bias)

    self.W_out = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.dropout = dropout

    self.mask = mask
    self.q_norm = q_norm
    self.k_norm = k_norm

  def forward(
    self,
    x: torch.Tensor,  # (B, T, C),
    mask=None,
  ):
    B, T, C = x.shape

    Q = self.Wq(x)
    K = self.Wk(x)
    V = self.Wv(x)

    if self.q_norm:
      Q = self.q_norm(Q)
    if self.k_norm:
      K = self.k_norm(K)

    if self.mask is not None:
      mask = self.mask
    elif self.mask == 'causal':
      mask = torch.ones(T, T, dtype=torch.bool)
      mask = torch.tril(mask)
    else:
      mask = None

    if self.kernel == 'sdpa':
      out = scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    else:
      attention = None
    out = self.W_out(out)

    return out
