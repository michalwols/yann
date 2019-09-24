
# yann (Yet Another Neural Network Library)

Yann is an extended version of torch.nn, adding a ton of sugar to make training models as fast and easy as possible.

## Getting Started

```shell script
pip install yann
```


### Training an MNIST Model

```python
import torch
from torch import nn
from torchvision import transforms

import yann
from yann.train import Trainer
from yann.modules import Stack, Flatten, Infer
from yann.params import HyperParams, Choice, Range


class Params(yann.HyperParams):
  batch_size = 32
  epochs = 10
  optimizer: Choice(('SGD', 'Adam')) = 'SGD'
  learning_rate: Range(.01, .0001) = .001

  seed = 1

# parse command line arguments
params = Params.from_command(prompt=True)
params.validate()

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
lenet(torch.rand(1, 28, 28))

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
    dataset='MNIST',
    transform=transforms.ToTensor(),
    loss='nll_loss',
    metrics=('accuracy',),
    device=yann.default_device
  )

train(params.epochs)

# save checkpoint
train.checkpoint()

# plot the loss curve
train.history.plot()
```

Start training run

```bash
python train.py -bs=16
```


