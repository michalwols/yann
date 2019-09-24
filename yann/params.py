

class HyperParams:
  # TODO:
  # track unused variables (warn)


  def __init__(self, **args):
    self.__dict__.update(args)

  def validate(self):
    pass

  @classmethod
  def from_command(cls, cmd=None, prompt=False, resolve=False):
    pass

  @classmethod
  def from_yaml(cls, path):
    pass

  def prompt(self, *fields):
    pass

  def on_change(self):
    pass

  def on_lookup(self):
    pass

  def __setattr__(self, key, value):
    raise AttributeError('Updating properties is not permitted')

  def __getitem__(self, item):
    if isinstance(item, (tuple, list)):
      return tuple(self.__dict__[k] for k in item)
    return self.__dict__[item]

  def fork(self, **args):
    return HyperParams(**{**self.__dict__, **args})

  def __repr__(self):
    return (
      'HyperParams('
      f"{', '.join(f'{k}={v}' for k, v in self.__dict__.items())}"
      ')'
    )

  def __str__(self):
    return (
        'HyperParams (\n' +
        ''.join('  {}={}\n'.format(k, v) for k, v in self.__dict__.items()) +
        ')'
    )

  def __len__(self):
    return len(self.__dict__)

  def __hash__(self):
    return hash(tuple(sorted(self.items())))

  def keys(self):
    return self.__dict__.keys()

  def values(self):
    return self.__dict__.values()

  def items(self):
    return self.__dict__.items()

  def grid(self, **args):
    raise NotImplementedError()

  def inject(self, scope=None, uppercase=True):
    scope = globals() if scope is None else scope
    for k, v in self.items():
      scope[k.upper() if uppercase else k] = v

  @classmethod
  def collect(cls, scope=None, types=(int, str, float, bool),
              upper_only=True, lowercase=True):
    scope = globals() if scope is None else scope

    d = {}
    for k, v in scope.items():
      if types and not isinstance(v, types):
        continue
      if upper_only and not k.isupper():
        continue

      d[k.lower() if lowercase else k] = v

    return cls(**d)

  def __eq__(self, other):
    return hash(self) == hash(other)


class Field:
  def __init__(self, help=None, required=True, none=False):
    self.help = help

class Choices(Field):
  pass

class Range(Field):
  def __init__(self, min, max):
    super(Range, self).__init__()
    self.min = min
    self.max = max

  def validate(self, val):
    assert self.min <= val <= self.max