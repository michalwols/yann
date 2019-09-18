from torch import nn

from ...data import Classes
from typing import Callable



class CNN(nn.Module):
  input_shape = (None, None, None, None)
  backbone: Callable

  def initialize(self):
    pass


class RecognitionModel(nn.Module):
  input_shape = (None, None, None, None)
  classes: Classes


  backbone: Callable
  # "Global" pooling layer that converts features into an embedding
  pool_features: Callable

  classifier: Callable


  def get_features(self, inputs):
    raise NotImplementedError()

  def get_embeddings(self, inputs):
    raise NotImplementedError()

  def get_logits(self, inputs):
    raise NotImplementedError()

  def freeze(self):
    pass

  def unfreeze(self):
    pass

  def freeze_backbone(self):
    pass

  def replace_classifier(self, linear=None, num_classes=None, init=None):
    raise NotImplementedError()