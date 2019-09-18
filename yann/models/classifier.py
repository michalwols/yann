from torch import nn

from ..data import Classes


class Classifier(nn.Module):
  classes: Classes

  def __del__(self, cls):
    raise NotImplementedError()

  def add_class(self, name, weight=None, bias=None):
    # TODO: copy other class, average multiple classes
    raise NotImplementedError()

  def remove_class(self, name):
    raise NotImplementedError()



class LinearClassifier(Classifier):
  def __init__(
      self,
      in_features,
      classes,
      bias=True,
      activation=None,
      test_activation=None
  ):
    super(Classifier, self).__init__()

    self.linear = nn.Linear(in_features, len(classes), bias=bias)
    self.classes = classes

    self.activation = activation
    self.test_activation = test_activation or activation