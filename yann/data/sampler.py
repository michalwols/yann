from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict


class BalancedTargetSampler(Sampler):
  def __init__(self, dataset):
    super().__init__(dataset)
    self.dataset = dataset

    self.target_to_indices = defaultdict(list)
    for n, targets in enumerate(self.dataset.targets):
      for t in targets:
        self.target_to_indices[t].append(n)
    self.targets = list(self.target_to_indices)

  def __iter__(self):
    for n in range(len(self)):
      t = random.choice(self.targets)
      yield random.choice(self.target_to_indices[t])

  def __len__(self):
    return len(self.dataset)