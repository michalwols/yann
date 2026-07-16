"""
Hyperparameter configuration, provided by the standalone ``hp`` package.

``yann.params`` is now a thin compatibility layer over ``hp``
(https://github.com/michalwols/hp). New code should import from ``hp``
directly; ``HyperParams`` here adds yann's legacy ``from_command`` entry
point on top of ``hp.HP``.
"""

from hp import (
  HP,
  Choice,
  Field,
  IntRange,
  LogIntRange,
  LogRange,
  Range,
  ValidationError,
  fields_from_callable,
  schema,
  wrap,
)


class HyperParams(HP):
  @classmethod
  def from_command(cls, cmd=None, validate=False, **kwargs):
    params = super().from_command(cmd)
    if validate:
      params.validate()
    return params

  def __getitem__(self, key):
    # legacy multi-key access: params['a', 'b'] == (params.a, params.b)
    if isinstance(key, (tuple, list)):
      return tuple(self[k] for k in key)
    return super().__getitem__(key)


__all__ = [
  'HP',
  'HyperParams',
  'Field',
  'Choice',
  'Range',
  'LogRange',
  'IntRange',
  'LogIntRange',
  'ValidationError',
  'schema',
  'wrap',
  'fields_from_callable',
]
