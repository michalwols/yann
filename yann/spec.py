class Spec:
  def __init__(self, *args, **kwargs):
    pass

  def state_dict(self):
    pass

  def check(self, *args, **kwargs):
    raise NotImplementedError()

  @classmethod
  def infer(cls):
    pass


class ModelSpec(Spec):
  def __init__(self, inputs, outputs, targets):
    super().__init__()


class Field:
  def __init__(
      self,
      name=None,
      description=None,
      required=True,
      default=None,
      validate=None,
      **props):
    self.name = name
    self.description = description
    self.required = required
    self.default = default
    self.validate = validate
    self.__dict__.update(props)


class Tensor(Field):
  pass


class Image(Tensor):
  pass


class Classes(Field):
  pass


class OneHot(Classes):
  pass


Resnet = lambda x: x

Resnet.spec = ModelSpec(
  inputs=Spec(
    Image(
      shape=(3, 224, 224),
      range=(0, 1),
      space='RGB',
      mean=(0.5, 0.5, 0.5),
      std=(0.5, 0.5, 0.5),
      dims=('channel', 'row', 'col'),
      dtype=None,
    )
  ),
  outputs=Spec(

  ),
  targets=Spec(
    OneHot(1000)
  ),
)


def adapt(src: Field, dest: Field):
  pass
