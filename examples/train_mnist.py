import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import yann
from yann.callbacks import History
from yann.callbacks.rich_progress import RichProgress
from yann.modules import Flatten, Infer, Stack
from yann.params import Choice, HyperParams, Range
from yann.train import Trainer


class Params(HyperParams):
  dataset = 'MNIST'
  batch_size = 32
  epochs = 10
  optimizer: Choice(('SGD', 'Adam')) = 'SGD'
  learning_rate: Range(0.01, 0.0001) = 0.01
  momentum = 0

  seed = 1


#
#
# class Vit(nn.Module):
#   def __init__(self):
#     self.conv = nn.Sequential(
#       nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#     )
#
#     self.transformers = nn.Sequential(
#       nn.TransformerEncoder()
#
#     )
#


class BoundedLeakyReLU(nn.Module):
  def __init__(self, negative_slope=0.01, peak=1):
    super(BoundedLeakyReLU, self).__init__()
    self.negative_slope = negative_slope
    self.peak = peak

  def forward(self, x):
    # Create the mask for the different regions
    positive_mask = (x > self.peak).float()  # x > 1 -> apply -x
    relu_mask = (x > 0).float() * (x <= self.peak).float()  # 0 <= x <= 1 -> apply ReLU
    leaky_relu_mask = (x <= 0).float()  # x <= 0 -> apply Leaky ReLU

    # Compute the outputs for each region
    negative_part = leaky_relu_mask * self.negative_slope * x
    relu_part = relu_mask * x
    inverted_part = positive_mask * (-x)

    # Combine the outputs
    return negative_part + relu_part + inverted_part


if __name__ == '__main__':
  # parse command line arguments
  params = Params.from_command()
  params.validate()

  print(params)

  # set random, numpy and pytorch seeds in one call
  # yann.seed(params.seed)

  Activation = BoundedLeakyReLU

  lenet = Stack(
    Infer(nn.Conv2d, 10, kernel_size=5),
    nn.MaxPool2d(2),
    Activation(),
    Infer(nn.Conv2d, 20, kernel_size=5),
    nn.MaxPool2d(2),
    Activation(),
    Flatten(),
    Infer(nn.Linear, 50),
    Activation(),
    Infer(nn.Linear, 10),
    activation=nn.LogSoftmax(dim=1),
  )

  # run a forward pass to infer input shapes using `Infer` modules
  lenet(torch.rand(1, 1, 28, 28))

  # use the registry to resolve optimizer name to an optimizer class
  optimizer = yann.resolve.optimizer(
    params.optimizer,
    yann.trainable(lenet.parameters()),
    momentum=params.momentum,
    lr=params.learning_rate,
  )

  train = Trainer(
    params=params,
    model=lenet,
    optimizer=optimizer,
    dataset=params.dataset,
    val_dataset=(params.dataset, {'train': False}),
    batch_size=params.batch_size,
    transform=transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
    ),
    loss='nll_loss',
    metrics=('accuracy',),
    callbacks=[
      # History("accuracy"),
      RichProgress(),
    ],
    amp=False,
  )

  train(params.epochs)

  # save checkpoint
  train.checkpoint()

  # plot the loss curve
  # train.history.plot()
