
from collections import defaultdict
import operator
from functools import reduce


class InvertedIndex:
  def __init__(self, items):
    self._index = defaultdict(set)
    self.update(items)

  def update(self, items):
    for key, vals in items:
      for v in vals:
        self._index[v].add(key)

  def __getitem__(self, vals):
    if isinstance(vals, tuple):
      return set(reduce(operator.and_, (self._index[v] for v in vals)))
    else:
      return self._index[vals]

  def get(self, vals, not_vals=None):
    if not_vals:
      return self[vals] - self[not_vals]
    else:
      return self[vals]