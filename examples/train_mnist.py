import torch
from torch import nn
from torchvision import transforms

import yann
from yann.modules import Stack, Flatten, Infer
from yann.params import HyperParams, Choice, Range
from yann.train import Trainer


class Params(HyperParams):
  dataset = 'MNIST'
  batch_size = 32
  epochs = 10
  optimizer: Choice(('SGD', 'Adam')) = 'SGD'
  learning_rate: Range(.01, .0001) = .01
  momentum = 0

  seed = 1

# parse command line arguments
params = Params.from_command()
params.validate()

print(params)

# set random, numpy and pytorch seeds in one call
yann.seed(params.seed)


lenet = Stack(
  Infer(nn.Conv2d, 10, kernel_size=5),
  nn.MaxPool2d(2),
  nn.ReLU(inplace=True),

  Infer(nn.Conv2d, 20, kernel_size=5),
  nn.MaxPool2d(2),
  nn.ReLU(inplace=True),

  Flatten(),

  Infer(nn.Linear, 50),
  nn.ReLU(inplace=True),
  Infer(nn.Linear, 10),
  activation=nn.LogSoftmax(dim=1)
)

# run a forward pass to infer input shapes using `Infer` modules
lenet(torch.rand(1, 1, 28, 28))

# use the registry to resolve optimizer name to an optimizer class
optimizer = yann.resolve.optimizer(
  params.optimizer,
  yann.trainable(lenet.parameters()),
  momentum=params.momentum,
  lr=params.learning_rate
)

train = Trainer(
  model=lenet,
  optimizer=optimizer,
  dataset=params.dataset,
  batch_size=params.batch_size,
  transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ]),
  loss='nll_loss',
  metrics=('accuracy',)
)

train(params.epochs)

# save checkpoint
train.checkpoint()

# plot the loss curve
train.history.plot()