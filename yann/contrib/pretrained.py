from yann.data.containers import Outputs, Inputs
from yann.models import Model


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
  def __init__(self, model, activation=None):
    super().__init__()
    self.model = model
    self.activation = activation

  def predict(self, inputs):
    if isinstance(inputs, Inputs):
      embeddings = self.model.features(*inputs.args, **inputs.kwargs)
    else:
      embeddings = self.model.features(inputs)
    logits = self.model.logits(embeddings)
    activations = self.activation(logits) if self.activation else None

    return Outputs(
      embeddings=embeddings,
      logits=logits,
      activations=activations
    )
