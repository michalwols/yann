from torch.utils.data.sampler import Sampler
import random
from collections import defaultdict

from . import batches


class BalancedTargetSampler(Sampler):
  def __init__(self, dataset, targets=None, size=None):
    super().__init__(dataset)
    self.dataset = dataset
    self.size = size

    self.target_to_indices = defaultdict(list)

    if not targets and hasattr(dataset, 'targets'):
      targets = dataset.targets

    if targets is not None:
      for n, ts in enumerate(targets):
        if isinstance(ts, (list, tuple)):
          for t in ts:
            self.target_to_indices[t].append(n)
        else:
          self.target_to_indices[ts].append(n)
    else:
      for n, (x, ts) in enumerate(dataset):
        if isinstance(ts, (list, tuple)):
          for t in ts:
            self.target_to_indices[t].append(n)
        else:
          self.target_to_indices[ts].append(n)

    self.targets = list(self.target_to_indices)

  def __iter__(self):
    for n in range(len(self)):
      # FIXME: use np.random to speed this up
      t = random.choice(self.targets)
      yield random.choice(self.target_to_indices[t])

  def __len__(self):
    return self.size or len(self.dataset)


from torch.utils.data import Sampler


class BatchShuffleSampler(Sampler):
  """
  Splits sorted data into `batch_size` chunks, then shuffles those chunks.

  Useful when dataset is sorted by length and want each batch to have items
  of around same length but want to randomize the batch order

  ex:
    list(BatchShuffleSampler(range(12), 3))
    >> [9, 10, 11, 3, 4, 5, 6, 7, 8, 0, 1, 2]
  """

  def __init__(self, data, batch_size):
    super().__init__(data)
    self.data = data
    self.batch_size = batch_size

  def __len__(self):
    return len(self.data)

  def __iter__(self):
    bs = list(batches(range(len(self)), size=self.batch_size))
    random.shuffle(bs)
    return (s for b in bs for s in b)