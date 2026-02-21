import torch


def normalize(batch, p=2, eps=1e-8):
  return batch / (
    torch.linalg.vector_norm(batch, p, dim=1, keepdim=True) + eps
  ).expand_as(
    batch,
  )
