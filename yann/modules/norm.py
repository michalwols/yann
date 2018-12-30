
import torch

def normalize(batch, p=2, eps=1e-8):
  return (
      batch
      / (torch.norm(batch, p=p, dim=1, keepdim=True) + eps).expand_as(batch)
  )