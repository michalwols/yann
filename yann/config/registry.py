from collections import defaultdict, OrderedDict

from functools import partial
from itertools import chain


def dedupe(items):
  return OrderedDict.fromkeys(items).keys()


def noop(x, *args, **kwargs):
  return x


def pass_args(x, *args, **kwargs):
  return x(*args, **kwargs)


def is_public(x):
  return (hasattr(x, '__name__') and not x.__name__.startswith('_'))


def is_public_callable(x):
  return (
      hasattr(x, '__name__')
      and not x.__name__.startswith('_')
      and callable(x)
  )

class default:
  init = pass_args

  @staticmethod
  def get_names(x):
    return x.__name__,


class RegistryError(Exception):
  pass


class ResolutionError(RegistryError):
  pass


class Record:
  __slots__ = ('x', 'init')

  def __init__(self, x, init=None):
    """
    Args:
      x: registered object or class
      init: optional function used to initialize the object
    """
    self.x = x
    self.init = init

  def __call__(self, *args, **kwargs):
    if self.init:
      return self.init(self.x, *args, **kwargs)
    else:
      return self.x(*args, **kwargs)


class Resolver:
  __slots__ = ('registry',)

  def __init__(self, registry: 'Registry'):
    self.registry = registry

  def resolve(
      self,
      x,
      required=False,
      validate=None,
      instance=True,
      types=None,
      init=None,
      args=None,
      kwargs=None
  ):
    initial = x
    if isinstance(x, str):
      record = self.registry[x]
      x = record.x
      if record.init:
        x = record.init(x, *(args or ()), **(kwargs or {}))

    if instance and isinstance(x, type):
      init = init or default.init
      x = init(x, *(args or ()), **(kwargs or {}))

    if not required and x is None:
      return x
    elif required and x is None:
      raise ResolutionError("Could not resolve to a value and was required")

    if types and not isinstance(x, types):
      raise ResolutionError(
        f"Failed to resolve {initial} to one of "
        f"{' '.join(str(t) for t in types)}, "
        f"got {x} instead of type {type(x)}"
      )

    if validate and not validate(x):
      raise ResolutionError(f"Couldn't validate {x}")

    return x

  def __call__(self, x, *_args, required=False, validate=None,
               instance=True, types=None, args=None, kwargs=None, init=None, **_kwargs):
    return self.resolve(
      x,
      required=required,
      validate=validate,
      instance=instance,
      types=types,
      args=args or _args,
      kwargs=kwargs or _kwargs,
      init=init
    )

  def __getattr__(self, name):
    registry = getattr(self.registry, name)
    return registry.resolve


class Registry:
  def __init__(self, types=None, private=False):
    """

    Args:
      types: only allow given types as entries
      private: if true will not have it's entries exposed to higher level registries
    """
    self._records = OrderedDict()
    self._subregistries = defaultdict(Registry)

    self.resolve = Resolver(self)

    self.types = types
    self.is_private = False

  def register(self, x, name=None, init=None):
    if isinstance(x, str) and name is None and init is None:
      # assuming usage as decorator, like:
      # @register("ReLu")
      # def relu():
      #   ...
      return partial(self.register, name=x)

    if self.types and not (
        issubclass(x, self.types) if isinstance(x, type)
        else isinstance(x, self.types)):
      raise RegistryError(
        f"Can't register an object of type {type(x)} in "
        f"typed registry which expects one of {self.types}")

    r = Record(x, init=init)

    if name is None:
      name = default.get_names(x)
    if isinstance(name, str):
      name = (name,)

    for n in name:
      self._records[n] = r

    return x

  def public_subregistries(self):
    for registry in self._subregistries.values():
      if not registry.is_private:
        yield registry


  def __contains__(self, name):
    if name in self._records:
      return True
    for registry in self.public_subregistries():
      if name in registry:
        return True
    return False

  def has(self, x):
    """
    Checks if this registry or one of its children contain the value x
    """
    return x in self.values()

  def __call__(self, x, name=None, init=None):
    return self.register(x, name, init)

  def __setitem__(self, name, x):
    if isinstance(x, Record):
      self._records[name] = x
    else:
      self.register(x=x, name=name)

  def __getattr__(self, name: str) -> 'Registry':
    if name in self.__dict__:
      return self.__dict__[name]
    return self._subregistries[name]

  def __setattr__(self, key, value):
    if isinstance(value, Registry):
      self._subregistries[key] = value
    else:
      self.__dict__[key] = value

  def __getitem__(self, item) -> Record:
    if item in self._records:
      return self._records[item]

    for registry in self.public_subregistries():
      try:
        return registry[item]
      except KeyError:
        pass

    raise KeyError(
      f"Couldn't find key: '{item}', "
      f"valid options include: {', '.join(self._records.keys())}")

  def values(self):
    return dedupe((
      *(r.x for r in self._records.values()),
      *chain(*(c.values() for c in self.public_subregistries()))
    ))

  def items(self):
    return (
      *((k, r.x) for k, r in self._records.items()),
      *chain(*(c.items() for c in self.public_subregistries()))
    )

  def keys(self):
    return (x[0] for x in self.items())

  def __len__(self):
    return len(self.values())

  def index(
      self,
      module,
      types=None,
      get_names=None,
      include=None,
      exclude=None,
      init=None,
      include_private=False
  ):
    """
    Indexes a module. If types are specified will only include entries of
    given type.
    Args:
      module:
      types:
      get_names:
      include:

    Returns:

    """
    for item in module.__dict__.values():
      if isinstance(item, (int, str)):
        continue
      if types:
        if not (
            isinstance(item, types) or
            (isinstance(item, type)) and issubclass(item, types)):
          continue

      if include and not include(item):
        continue

      if exclude and exclude(item):
        continue

      if not include_private and not is_public(item):
        continue


      names = get_names(item) if get_names else default.get_names(item)

      self.register(item, name=names, init=init)

  def register_subclasses(self, cls: type, init=None):
    self.register(cls, init=init)
    for scls in cls.__subclasses__():
      self.register(scls, init=init)

  def update(self, items, init=None):
    for x in items:
      self.register(x, init=init)

  def print_tree(self, contents=True, indent=0):
    if not indent:
      print(f'registry{" (Private - not resolvable from higher scopes)" if self.is_private else ""}')
      indent += 2
    for name, registry in self._subregistries.items():
      print(f"{' ' * indent}.{name} {' (Private - not resolvable from higher scopes)' if registry.is_private else ''}")
      registry.print_tree(indent=indent + 2, contents=contents)

    if contents:
      for name, record in self._records.items():
        if isinstance(record.x, partial) or not hasattr(record.x, '__module__'):
          details = str(record.x)
        else:
          details = f"{record.x.__module__}.{record.x.__name__}"
        print(f"{' ' * (indent + 2) }- {name}\t\t({details})")


  def __str__(self):
    parts = [
      ''.join(f"  '{name}': {r.x.__name__}  ({r.init})\n"
              for name, r in self._records.items())
    ]

    for name, c in self._subregistries.items():
      parts.append(f"\n{name}:\n")
      parts.append(str(c))

    return ''.join(parts)
