from __future__ import annotations

from typing import Any, Callable


def remote(fn: Callable | None = None, **options: Any):
  try:
    import ray
  except ImportError as error:
    raise ImportError('install yann[ray] to use Ray helpers') from error

  def apply(target):
    return ray.remote(**options)(target) if options else ray.remote(target)

  return apply(fn) if fn is not None else apply


def map_trials(fn: Callable, trials, **options: Any):
  worker = remote(fn, **options)
  return [worker.remote(trial) for trial in trials]
