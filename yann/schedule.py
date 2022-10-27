
from functools import update_wrapper
from typing import Any



class Scheduler:
  def __init__(self):
    self.index = 0

  def step(self):
    self.index += 1

  def state_dict(self):
    return self.__dict__

  def load_state_dict(self, d):
    self.__dict__.update(d)


def schedule(index: int) -> Any:
  pass


class Scheduled:
  def __init__(self, func, get_step=None, **params):
    self.func = func
    self.params = params

    self.get_step = get_step
    self.call_count = 0

    update_wrapper(self, func)

  def __call__(self, *args, **kwargs):
    step = self.get_step() if self.get_step else self.call_count

    for param, get_value in self.params.items():
      kwargs[param] = get_value(step)

    self.call_count += 1
    return self.func(*args, **kwargs)


def scheduled(*, get_step=None, **params):
  """
  A functools.partial like decorator that adjusts the partial values using
  schedule functions

  Ex:
    @scheduled(scale=lambda n: n * 2)
    def transform(image, scale=None):
      print(scale)

  Args:
    get_step: optional callable to use to determine current step
    **params: mapping from decorated function argument to a `schedule` function
      which take current step and returns a new value

  Returns:

  """
  def decorator(func):
    return Scheduled(func, **params, get_step=get_step)

  return decorator



