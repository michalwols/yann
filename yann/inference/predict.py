import torch

from .. import evalmode


class Predictor:
  def __call__(self, *args, **kwargs):
    x = self.load(*args, **kwargs)
    x = self.transform(x)
    x = self.predict(x)
    x = self.posprocess(x)

    return x

  def export(self):
    pass

  def load(self, samples):
    return samples

  def transform(self, inputs):
    return inputs

  def predict(self, inputs):
    return inputs

  def posprocess(self, inputs):
    return inputs


class Classifier(Predictor):
  def __init__(self, model, classes=None, preprocess=None, postprocess=None):
    self.model: torch.nn.Module = model
    self.preprocess = preprocess
    self.postprocess = postprocess

    self.classes = classes

  def __call__(self, *args, **kwargs):
    x = self.preprocess(*args, **kwargs)
    with evalmode(self.model):
      x = self.model(x)

    return self.classes.decode(x)
