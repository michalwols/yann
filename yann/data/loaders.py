from torch.utils.data import DataLoader

from .datasets import TransformDataset


class TransformLoader(DataLoader):
  def __init__(self, dataset, transforms, **kwargs):
    super(TransformLoader, self).__init__(
      TransformDataset(dataset,  transforms),
      **kwargs
    )