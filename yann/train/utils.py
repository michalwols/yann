from torch import nn


def replace_last_linear(model, num_classes, layer_name='last_linear'):
  setattr(
    model,
    layer_name,
    nn.Linear(
      getattr(model, layer_name).in_features,
      num_classes
    )
  )