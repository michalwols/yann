import torch
from pretrainedmodels.models import squeezenet1_1
from torch import nn

import yann
from yann.contrib.pretrained import PretrainedModelWrapper
from yann.utils.timer import Timer


def test_pretrained():
  timer = Timer(log=True)

  with timer.task('initialize model'):
    model = squeezenet1_1(1000, pretrained=None)

  wrapped = PretrainedModelWrapper(model, activation=nn.Softmax())

  inputs = torch.rand(1, 3, 255, 255)

  with yann.evalmode(wrapped), timer.task('model predict'):
    outputs = wrapped.predict(inputs)
  with yann.evalmode(model), timer.task('model predict'):
    model_outputs = model(inputs)

  assert torch.all(outputs.logits == model_outputs)
  assert outputs.logits.shape
  assert outputs.embeddings.shape
  assert outputs.activations.shape
