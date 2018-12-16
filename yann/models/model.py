from torch import nn

from ..data.containers import Outputs


class ModelMixin:
  def __init__(self, *args, **kwargs):
    # Need to store this for when we save the model,
    # so that we can restore it later without knowing them
    self._init_args = {
      'args': args,
      'kwargs': kwargs
    }
    super().__init__(*args, **kwargs)

  def predict(self, inputs) -> Outputs:
    """
    Inference mode predict call
    """
    return None

  @classmethod
  def load(cls, *args, **kwargs) -> 'Model':
    return cls(*args, **kwargs)


class Model(ModelMixin, nn.Module):
  pass
