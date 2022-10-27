from collections import defaultdict, Counter
from itertools import chain


def by(items, attr=None, key=None, unique=False, dest=None):
  if attr:
    if unique:
      return {getattr(x, attr): x for x in items}
    else:
      d = dest or defaultdict(list)
      for x in items:
        d[getattr(x, attr)].append(x)
      return d
  if key:
    if unique:
      return {x[key]: x for x in items}
    else:
      d = dest or defaultdict(list)
      for x in items:
        d[x[key]].append(x)
      return d


def count(*args):
  return Counter(chain(*args))



class Collection:
  def __init__(self, items):
    self.items = items

  def __iter__(self):
    return iter(self.items)

  def __len__(self):
    return len(self.items)

  def __getitem__(self, item):
    return self.items[item]

  def values(self, *attrs, flat=True):
    if len(attrs) == 1 and flat:
      a = attrs[0]
      yield from (getattr(x, a) for x in self.items)
    else:
      yield from (tuple(getattr(x, a) for a in attrs) for x in self.items)

  def filter(self, condition):
    return Collection(
      x for x in self if condition(x)
    )

  def map(self, f):
    return Collection(f(x) for x in self)

  def sorted(self, *props, reverse=False, key=None):
    if props:
      return sorted(
        self.items,
        key=lambda x: tuple(getattr(x, p) for p in props),
        reverse=reverse
      )

    return sorted(
      self.items,
      key=key,
      reverse=reverse
    )

  def __getattr__(self, name: str):
    if name.startswith('by_unique_'):
      x = by(self.items, name[len('by_unique_'):], unique=True)
      setattr(self, name, x)
      return x

    if name.startswith('by_'):
      x = by(self.items, name[3:])
      setattr(self, name, x)
      return x

    if name.endswith('_counts'):
      attr = name[:-len('_counts')]
      x = count(getattr(x, attr) for x in self.items)
      setattr(self, name, x)
      return x

    if name.endswith('_set'):
      attr = name[:-len('_set')]
      x = set(getattr(x, attr) for x in self.items)
      setattr(self, name, x)
      return x

    if '_to_' in name:
      src, dst = name.split('_to_')
      x = {getattr(x, src): getattr(x, dst) for x in self.items}
      setattr(self, name, x)
      return x

    raise AttributeError(name)

