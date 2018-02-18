

class Config:
  """
  TODO:
  - back by environment variables
  - set defaults
  - set required values
  - serialize
  - dict like object
  """

  def __init__(self, defs=None):
    self.values = dict(defs)

  def load(self, module, types='primitives', keep=None):
    """loads all variables from given module"""
    pass

  def capture(self, keep=lambda k, v: True):
    for k, v in globals().items():
      if keep(k, v):
        self.values[k] = v


  def expose(self, overwrite=True):
    g = globals()
    for k, v in self.values.items():
      if not overwrite and k in g:
        continue
      g[k] = v

  def track(self, inputs=True, outputs=True, context=None):
    pass

c = Config()
c.lr = 3
c.num_epochs = 4


class Env:
  """TODO: capture running environment"""