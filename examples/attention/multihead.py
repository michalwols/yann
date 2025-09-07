import torch


def multihead_attention(X_btd, mask, n_heads, W_kqv, W_out):
  dB, dT, dD = X_btd.shape
  K, Q, V = torch.split(X_btd @ W_kqv, 3, dim=-1)

  # B x T x D => B x heads x T x d/heads
  K.reshape()
