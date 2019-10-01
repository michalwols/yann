from time import time as get_time


def padded_insert(items, index, value, null_val=None):
  """
  insert value into the items list at given index
  if index is larger than length of list then
  extend it up to index and pad the extra space with null_val
  """
  if len(items) == index:
    items.append(value)
  elif len(items) > index:
    items[index] = value
  else:
    items.extend([null_val] * (index - len(items)))
    items.append(value)
  return items


class PaddedList(list):
  def __init__(self, *args, null_val=None):
    super().__init__(*args)
    self.null_val = null_val

  def __setitem__(self, index, value):
    if len(self) == index:
      self.append(value)
    elif len(self) > index:
      super().__setitem__(index, value)
    else:
      self.extend([self.null_val] * (index - len(self)))
      self.append(value)


class MetricStore:
  def __init__(self, names=None, null_val=None, cast_value=None):
    self.values = {n: PaddedList(null_val=null_val) for n in names} if names else {}
    self.times = PaddedList(null_val=null_val)

    self.null_val = null_val
    self.cast_value = cast_value

  def update(self, step='next', time='now', **values):
    step = len(self.times) if step == 'next' else step
    time = get_time() if time == 'now' else time

    if self.cast_value:
      values = {k: self.cast_value(v) for (k, v) in values.items()}

    self.times[step] = time
    for metric, v in values.items():
      if metric not in self.values:
        self._init_metric(metric)
      self.values[metric][step] = v

    # need to make sure all metrics have the same length
    for metric, vs in self.values.items():
      if len(vs) < len(self.times):
        vs[len(self.times) - 1] = self.null_val

  def _init_metric(self, name):
    self.values[name] = PaddedList(null_val=self.null_val)

  def __getitem__(self, name):
    return self.values[name]

  def __contains__(self, item):
    return item in self.values

  def keys(self):
    return self.values.keys()

  def items(self):
    return self.values.items()

  def __len__(self):
    return len(self.times)

  def to_pandas(self):
    import pandas as pd
    return pd.DataFrame({'times': self.times, **self.values})

  def plot(self, metrics=None, time=False, clear=False):
    if clear:
      try:
        from IPython.display import clear_output
        clear_output(wait=True)
      except:
        pass

  def __repr__(self):
    return  f"MetricStore({', '.join(f'{k}=(min={min(v)}, max={max(v)})' for k,v in self.values.items())}, len={len(self)})"


class EventStore(list):
  def add(self, *args, key=None, value=None, step=None, time=None):
    self.append(Event(key=key, value=value, step=step, time=time))


class Event:
  __slots__ = ['key', 'value', 'step', 'time']

  def __init__(self, key=None, value=None, step=None, time=None):
    self.key = key
    self.value = value
    self.step = step
    self.time = time