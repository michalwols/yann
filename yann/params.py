"""
Hyperparameter configuration, provided by the standalone ``hp`` package.

``yann.params`` is a thin compatibility layer over ``hp``
(https://github.com/michalwols/hp). New code should import from ``hp``
directly.

``hp.Params`` deliberately has no public methods, so that every attribute name
stays available for user fields. Operations are module-level functions taking
the params first: ``hp.to_dict(params)``, ``hp.fork(params)`` and so on.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Dict

from hp import (
  Choice,
  Dynamic,
  Evolve,
  Field,
  IntRange,
  LogIntRange,
  LogRange,
  Params,
  Range,
  ValidationError,
  fields_from_callable,
  items,
  load,
  params,
  schema,
  to_dict,
  update,
  validate,
)
from hp import cli as _cli


class HyperParams(Params):
  """``hp.Params`` plus yann's legacy entry points.

  Only classmethods and dunders are added, so the instance namespace stays
  empty for user fields.
  """

  @classmethod
  def from_command(cls, cmd=None, validate_params=False, **kwargs):
    params = load(cls, _cli(cmd) if cmd is not None else _cli)
    if validate_params:
      validate(params)
    return params

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]):
    return cls(**dict(data))

  def __getitem__(self, key):
    # legacy multi-key access: params['a', 'b'] == (params.a, params.b)
    if isinstance(key, (tuple, list)):
      return tuple(self[k] for k in key)
    return super().__getitem__(key)


def inject(params: Params, scope=None, uppercase=True):
  """Copy params into a namespace, uppercased by default."""
  scope = globals() if scope is None else scope
  for key, value in items(params):
    scope[key.upper() if uppercase else key] = value


def collect(
  cls,
  scope=None,
  types=(int, str, float, bool),
  upper_only=True,
  lowercase=True,
):
  """Build params from the constants in a namespace."""
  scope = globals() if scope is None else scope

  values = {}
  for key, value in scope.items():
    if types and not isinstance(value, types):
      continue
    if upper_only and not key.isupper():
      continue
    values[key.lower() if lowercase else key] = value

  return cls(**values)


def save_params(params: Params, path):
  from hp import save

  save(params, path)


_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _serialize_param_value(value, *, _depth=0):
  if isinstance(value, _PRIMITIVE_TYPES):
    return value

  if isinstance(value, Params):
    return to_serializable_dict(value)

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


def to_serializable_dict(params: Params) -> Dict[str, Any]:
  """Params as a dict safe to serialize, stringifying unsupported objects."""
  return {k: _serialize_param_value(v) for k, v in items(params)}


__all__ = [
  'Params',
  'HyperParams',
  'Dynamic',
  'Field',
  'Choice',
  'Range',
  'LogRange',
  'IntRange',
  'LogIntRange',
  'Evolve',
  'ValidationError',
  'schema',
  'params',
  'load',
  'fields_from_callable',
  'to_dict',
  'items',
  'update',
  'validate',
  'inject',
  'collect',
  'save_params',
  'to_serializable_dict',
]
