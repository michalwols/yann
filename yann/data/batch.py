from dataclasses import dataclass
from typing import Any, Optional

from yann import to


@dataclass
class Batch:
  ids: Optional[Any] = None
  inputs: Optional[Any] = None
  targets: Optional[Any] = None
  outputs: Optional[Any] = None
  losses: Optional[Any] = None

  @property
  def size(self):
    if self.inputs is not None:
      return len(self.inputs)
    raise ValueError('Could not determine size')

  def __iter__(self):
    yield self.inputs
    yield self.outputs

  def to(self, device, **kwargs):
    to(
      self.inputs,
      self.targets,
      self.outputs,
      self.losses,
      device=device,
      **kwargs,
    )
    return self


batch = Batch()
batch.size

for batch in batches:
  batch.to('cuda:0')


class Batch:
  inputs: Inputs
  targets: Targets

  def __init__(self, inputs, targets, outputs=None):
    self.inputs = inputs
    self.targets = targets

  def __iter__(self):
    return (*self.inputs, *self.targets)

  def to(self, device):
    pass

  @property
  def size(self):
    return len(self.inputs[0])

  def items(self):
    return []


class DetectionBatch(Batch):
  boxes: Inputs


class SegmentationBatch(Batch):
  masks: Inputs


batches: List[Batch] = []
for b in batches:
  b.to(model.device)
  inputs, targets = b

  outputs = model(*inputs)
  b.outputs = outputs
