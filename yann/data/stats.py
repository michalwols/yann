import torch

class TensorStats:
  def __init__(self, device=None, dim=0):
    self.total = None
    self.min = None
    self.max = None

    self.count = None

    self.device = device
    self.dim = dim

  @torch.inference_mode()
  def update(self, batch: torch.Tensor):
    if self.device:
      batch = batch.to(self.device)
    sum = batch.sum(dim=self.dim)

    if self.total is None:
      self.total = sum
      self.count = batch.shape[self.dim]
      self.max = torch.max(batch, dim=self.dim)[0]
      self.min = torch.min(batch, dim=self.dim)[0]
    else:
      self.total += sum
      self.count += batch.shape[self.dim]
      self.max = torch.maximum(self.max, batch.max(dim=self.dim)[0], out=self.max)
      self.min = torch.minimum(self.min, batch.min(dim=self.dim)[0], out=self.min)

  @property
  def mean(self):
    return self.total / self.count
