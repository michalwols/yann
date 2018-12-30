import torch
from torch.utils.data.dataloader import default_collate
from .utils import pad_to_largest


class PadCollate:
  def __init__(self, pad_inputs=True, pad_targets=False, value=0):
    self.pad_inputs = pad_inputs
    self.pad_targets = pad_targets
    self.value = value

  def __call__(self, batch):
    inputs, targets = zip(*batch)

    if self.pad_inputs:
      inputs = pad_to_largest(list(inputs), self.value)
    if self.pad_targets:
      targets = pad_to_largest(list(targets), self.value)

    return torch.stack(inputs), torch.stack(targets)



class FilterCollate:
  def __init__(self, filter=None, collate=default_collate, value=None):
    self.filter = filter or (
      lambda items: [it for it in items if it is not value]
    )
    self.collate = collate

  def __call__(self, batch):
    filtered = self.filter(batch)
    return self.collate(filtered)


filter_collate = FilterCollate()


