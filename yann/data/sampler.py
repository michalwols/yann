from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict


class BalancedTargetSampler(Sampler):
  def __init__(self, dataset, targets=None, size=None):
    super().__init__(dataset)
    self.dataset = dataset
    self.size = size

    self.target_to_indices = defaultdict(list)

    if not targets and hasattr(dataset, 'targets'):
      targets = dataset.targets

    if targets:
      for n, ts in enumerate(targets):
        for t in ts:
          self.target_to_indices[t].append(n)
    else:
      for n, (x, ts) in enumerate(dataset):
        for t in ts:
          self.target_to_indices[t].append(n)

    self.targets = list(self.target_to_indices)

  def __iter__(self):
    for n in range(len(self)):
      t = random.choice(self.targets)
      yield random.choice(self.target_to_indices[t])

  def __len__(self):
    return self.size or len(self.dataset)