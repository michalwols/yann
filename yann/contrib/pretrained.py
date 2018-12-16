from torch.nn import AdaptiveAvgPool2d
from ..data.containers import Outputs, Inputs
from ..models import Model


class PretrainedModel(Model):
  def __init__(self, *args, **kwargs):
    self.activation = None
    super().__init__(*args, **kwargs)

  def predict(self, inputs):
    embeddings = self.features(*inputs)
    logits = self.logits(embeddings)
    activations = self.activation(logits) if self.activation else None
    return Outputs(
      embeddings=embeddings,
      logits=logits,
      activations=activations
    )


class PretrainedModelWrapper(Model):
  def __init__(self, model, activation=None, pool=None):
    super().__init__()
    self.model = model
    self.activation = activation

    self.pool = pool or AdaptiveAvgPool2d(1)
    if self.pool:
      self.model.avg_pool = pool

    if hasattr(self.model, 'avgpool') and not hasattr(self.model, 'avg_pool'):
      # the pretrained model api is inconsistent and a few cases have avgpool
      # instead of avg_pool
      self.model.avg_pool = self.model.avgpool

  def predict(self, inputs):
    if isinstance(inputs, Inputs):
      features = self.model.features(*inputs.args, **inputs.kwargs)
    else:
      features = self.model.features(inputs)

    print(features.shape)

    # pretrained models return feature maps before pooling and reshaping
    if self.pool:
      embeddings = (self.pool(features)).view(features.shape[0], -1)
    else:
      embeddings = (self.model.avg_pool(features)).view(features.shape[0], -1)

    logits = self.model.logits(features)
    activations = self.activation(logits) if self.activation else None

    return Outputs(
      embeddings=embeddings,
      logits=logits,
      activations=activations
    )


def spec_from_settings():
  pass


def spec_from_model(model):
  pass
