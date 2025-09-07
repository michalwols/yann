from typing import Unpack

import torch.nn.functional as F
from torch import nn

from yann import params
from yann.modules import Residual, Stack
from yann.modules.activations import relu_squared
from yann.train import Trainer


class Params(Trainer.Params):
  norm = 'rmsnorm'

  ffn = 'swiglu'

  patch_size = 8
  patch_mode = 'conv'

  registers = None

  class_token = True

  positional_encoding = None

  embed_dim = 512
  depth = 6
  num_heads = 8
  mlp_ratio = 4

  sigmoid_attention = False


class SelfAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, bias=False, dropout=None):
    super().__init__()
    self.dropout = dropout
    self.embed_dim = embed_dim
    self.num_heads = num_heads

    self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
    self.Wout = nn.Linear(embed_dim, embed_dim, bias=bias)

  def forward(self, x):
    B, T, C = x.shape
    Q, K, V = self.Wqkv(x).chunk(3, dim=-1)

    #  (B, T, C) => (B, num_heads, T, head_dim)
    Q = Q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    K = K.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    V = V.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

    y = F.scaled_dot_product_attention(
      Q,
      K,
      V,
      dropout_p=self.dropout if self.training else 0,
      is_causal=False,
    )

    # (B, num_heads, T, head_dim) => (B, T, num_heads * head_dim)
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    return self.Wout(y)


class MLP(Stack):
  def __init__(self, in_channels, inner_channels):
    super().__init__(
      nn.Linear(in_channels, inner_channels),
      relu_squared,
      nn.Linear(inner_channels, in_channels),
    )


class ParamsMNIST(Params):
  dataset = 'MNIST'
  patch_size = 4


class SigmoidAttention(nn.Module):
  def __init__(self):
    super().__init__()

    self.wq = nn.Linear()
    self.wk = nn.Linear()
    self.wv = nn.Linear()


class ViT(nn.Module):
  def __init__(
    self,
  ):
    self.patchify = PatchEmbedding()
    self.blocks = Stack(
      Residual(nn.RMSNorm(), SelfAttention()),
      Residual(nn.RMSNorm(), MLP()),
    )


def get_trainer(self, params: Params):
  train = Trainer.from_params(params)
  return train
