import torch
from torch import nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
  """
  Multi head grouped query attention


  """

  def __init__(
    self,
    embed_dim,
    num_heads,
    num_groups,
    bias=False,
    dropout=None,
  ):
    super().__init__()

    self.embed_dim = embed_dim
    self.num_heads = num_heads

    self.num_groups = num_groups

    self.head_dim = embed_dim // num_heads
    self.group_dim = embed_dim // num_groups

    self.wq = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.wk = nn.Linear(embed_dim, self.group_dim, bias=bias)
    self.wv = nn.Linear(embed_dim, self.group_dim, bias=bias)

    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.dropout = dropout

  def forward(self, x):
    B, T, C = x.shape

    Q = self.wq(x)
    K = self.wk(x)
    V = self.wv(x)
