"""
Hyperparameter configuration, provided by the standalone ``hp`` package.

``yann.params`` is now a thin compatibility layer over ``hp``
(https://github.com/michalwols/hp). New code should import from ``hp``
directly; ``HyperParams`` here adds yann's legacy ``from_command`` entry
point on top of ``hp.HP``.
"""

import sys

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


def _cli_args(cmd=None):
  if isinstance(cmd, str):
    args = cmd.split()
  elif cmd is None:
    args = list(sys.argv[1:])
  else:
    args = list(cmd)

  # legacy argparse flags used dashes, hp paths use underscores
  normalized = []
  for arg in args:
    if arg.startswith('--'):
      key, eq, value = arg[2:].partition('=')
      arg = f'--{key.replace("-", "_")}{eq}{value}'
    normalized.append(arg)
  return normalized


class HyperParams(HP):
  @classmethod
  def from_command(cls, cmd=None, validate=False, **kwargs):
    params = cls.from_cli(_cli_args(cmd))
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
