from .train import BaseTrainer



def seed(val=None):
  import numpy as np
  import torch
  if val:
    torch.manual_seed(val)
    np.random.seed(val)
  return val
