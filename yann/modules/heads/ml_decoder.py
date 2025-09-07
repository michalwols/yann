import torch
from torch import nn

from yann.modules import Residual, Stack


class MLTransformerDecoderLayer(Stack):
  def __init__(
    self,
    embed_dim,
    num_heads=8,
    feedforward_dim=2048,
    dropout=0.1,
    layer_norm_eps=1e-5,
  ):
    super().__init__(
      norm1=Stack(
        Residual(nn.Dropout(dropout)),
        nn.LayerNorm(embed_dim, eps=layer_norm_eps),
      ),
      attention=nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout),
      norm2=Stack(
        Residual(nn.Dropout(dropout)),
        nn.LayerNorm(embed_dim, eps=layer_norm_eps),
      ),
      feedforward=Stack(
        nn.Linear(embed_dim, feedforward_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(feedforward_dim, embed_dim),
      ),
      norm3=Stack(
        Residual(nn.Dropout(dropout)),
        nn.LayerNorm(embed_dim, eps=layer_norm_eps),
      ),
    )

  def forward(self, x, memory, **kwargs):
    x = self.norm1(x)
    x = self.attention(x, memory, memory)[0]
    x = self.norm2(x)
    x = self.feedforward(x)
    return self.norm3(x)


@torch.jit.script
class GroupFC(object):
  def __init__(self, embed_len_decoder: int):
    self.embed_len_decoder = embed_len_decoder

  def __call__(
    self,
    h: torch.Tensor,
    duplicate_pooling: torch.Tensor,
    out_extrap: torch.Tensor,
  ):
    for i in range(self.embed_len_decoder):
      h_i = h[:, i, :]
      w_i = duplicate_pooling[i, :, :]
      out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class GroupFullyConnectedPooling(nn.Module):
  def __init__(self, num_classes, embed_len_decoder, decoder_embedding):
    super(GroupFullyConnectedPooling, self).__init__()

    self.num_classes = num_classes
    self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
    self.duplicate_pooling = torch.nn.Parameter(
      torch.Tensor(embed_len_decoder, decoder_embedding, self.duplicate_factor),
    )
    self.bias = torch.nn.Parameter(torch.Tensor(num_classes))
    self.group_fc = GroupFC(embed_len_decoder)

    self._init_weights()

  def _init_weights(self):
    torch.nn.init.xavier_normal_(self.duplicate_pooling)
    torch.nn.init.constant_(self.bias, 0)

  def forward(self, h):
    out_extrap = torch.zeros(
      h.shape[0],
      h.shape[1],
      self.duplicate_factor,
      device=h.device,
      dtype=h.dtype,
    )
    self.group_fc(h, self.duplicate_pooling, out_extrap)
    logits = out_extrap.flatten(1)[:, : self.num_classes]
    logits += self.bias
    return logits


class MLDecoder(nn.Module):
  def __init__(
    self,
    num_classes,
    num_groups=None,
    decoder_embed_dim=768,
    initial_num_features=2048,
    feedforward_dim=1024,
    num_heads=8,
    dropout=0.1,
  ):
    super().__init__()
    num_groups = num_groups or num_classes

    self.num_classes = num_classes
    self.num_groups = num_groups

    # switching to 768 initial embeddings
    decoder_embed_dim = 768 if decoder_embed_dim < 0 else decoder_embed_dim

    self.query_embed = nn.Embedding(num_groups, decoder_embed_dim)
    self.query_embed.requires_grad_(False)

    self.embed_input = Stack(
      nn.Linear(initial_num_features, decoder_embed_dim),
      nn.ReLU(inplace=True),
    )

    self.decoder = nn.TransformerDecoder(
      MLTransformerDecoderLayer(
        embed_dim=decoder_embed_dim,
        feedforward_dim=feedforward_dim,
        num_heads=num_heads,
        dropout=dropout,
      ),
      num_layers=1,
    )

    self.group_pooling = GroupFullyConnectedPooling(
      num_classes=num_classes,
      embed_len_decoder=num_groups,
      decoder_embedding=decoder_embed_dim,
    )

  def forward(self, x: torch.Tensor):
    if len(x.shape) == 4:  # [bs,2048, 7,7]
      x = x.flatten(2).transpose(1, 2)

    x = self.embed_input(x)
    # no allocation of memory with expand
    target = self.query_embed.weight.unsqueeze(1).expand(-1, x.shape[0], -1)
    h = self.decoder(
      target,
      x.transpose(0, 1),
    )  # [embed_len_decoder, batch, 768]
    h = h.transpose(0, 1)

    return self.group_pooling(h)
