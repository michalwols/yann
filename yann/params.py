"""
Hyperparameter configuration, provided by the standalone ``hp`` package.

``yann.params`` is now a thin compatibility layer over ``hp``
(https://github.com/michalwols/hp). New code should import from ``hp``
directly; ``HyperParams`` here adds yann's legacy entry points on top of
``hp.HP``.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Dict

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

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]):
    return cls(**dict(data))

  def __getitem__(self, key):
    # legacy multi-key access: params['a', 'b'] == (params.a, params.b)
    if isinstance(key, (tuple, list)):
      return tuple(self[k] for k in key)
    return super().__getitem__(key)

  def inject(self, scope=None, uppercase=True):
    scope = globals() if scope is None else scope
    for k, v in self.items():
      scope[k.upper() if uppercase else k] = v

  @classmethod
  def collect(
    cls,
    scope=None,
    types=(int, str, float, bool),
    upper_only=True,
    lowercase=True,
  ):
    scope = globals() if scope is None else scope

    values = {}
    for k, v in scope.items():
      if types and not isinstance(v, types):
        continue
      if upper_only and not k.isupper():
        continue
      values[k.lower() if lowercase else k] = v

    return cls(**values)


def to_dict(params: HP) -> Dict[str, Any]:
  return params.to_dict()


def save_params(params: HP, path):
  params.save(path)


_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _serialize_param_value(value, *, _depth=0):
  if isinstance(value, _PRIMITIVE_TYPES):
    return value

  if isinstance(value, Mapping):
    return {
      str(k): _serialize_param_value(v, _depth=_depth + 1) for k, v in value.items()
    }

  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    return [_serialize_param_value(v, _depth=_depth + 1) for v in value]

  if isinstance(value, set):
    return [_serialize_param_value(v, _depth=_depth + 1) for v in sorted(value, key=repr)]

  try:
    from yann.utils import fully_qualified_name

    return fully_qualified_name(value)
  except Exception:
    return repr(value)


def to_serializable_dict(params: HP) -> Dict[str, Any]:
  """Params as a dict safe to serialize, stringifying unsupported objects."""
  return {k: _serialize_param_value(v) for k, v in params.items()}


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
  'to_dict',
  'save_params',
  'to_serializable_dict',
]
