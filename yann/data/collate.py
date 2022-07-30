import numpy as np
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
  def __init__(
    self, filter=None, collate=default_collate, value=None
  ):
    self.filter = filter or (
      lambda items: [it for it in items if it is not value]
    )
    self.collate = collate

  def __call__(self, batch):
    filtered = self.filter(batch)
    return self.collate(filtered)


filter_collate = FilterCollate()


def image_collate(batch, memory_format=torch.contiguous_format):
  """
  Note: using tensor cores channels_last should lead to better performance
    inspired by: https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L28
  Args:
    batch: [(image, target), ...]
  """
  images, targets = zip(*batch)
  targets = torch.tensor(targets, dtype=torch.int64)

  w, h = images[0].shape[:2]

  tensor = torch.zeros(
    (len(images), 3, h, w),
    dtype=torch.uint8,
  ).contiguous(memory_format=memory_format)

  for i, img in enumerate(images):
    nump_array = np.asarray(img, dtype=np.uint8)
    if (nump_array.ndim < 3):
      nump_array = np.expand_dims(nump_array, axis=-1)
    nump_array = np.rollaxis(nump_array, 2)
    tensor[i] += torch.from_numpy(nump_array)
  return tensor, targets



class KeyCollate:
  def __init__(self, *keys):
    self.keys = keys

  def __call__(self, samples):
    return tuple(
      torch.stack(
        [s[k] for s in samples]
      ) for k in self.keys
    )