
# yann (Yet Another Neural Network Library)

Yann is an extended version of torch.nn, adding a ton of sugar to make training models as fast and easy as possible.

## Getting Started

### Install 

```shell script
pip install yann
```


### Train LeNet on MNIST

```python
import torch
from torch import nn
from torchvision import transforms

import yann
from yann.train import Trainer
from yann.modules import Stack, Flatten, Infer
from yann.params import HyperParams, Choice, Range


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
  metrics=('accuracy',),
  device=yann.default_device
)

train(params.epochs)

# save checkpoint
train.checkpoint()

# plot the loss curve
train.history.plot()
```

view the generated cli help
```bash
python train.py -h
```

```shell script
-h
usage: train_mnist.py [-h] [-o {SGD,Adam}] [-lr LEARNING_RATE] [-d DATASET]
                      [-bs BATCH_SIZE] [-e EPOCHS] [-m MOMENTUM] [-s SEED]

optional arguments:
  -h, --help            show this help message and exit
  -o {SGD,Adam}, --optimizer {SGD,Adam}
                        optimizer (default: SGD)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning_rate (default: 0.01)
  -d DATASET, --dataset DATASET
                        dataset (default: MNIST)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        epochs (default: 10)
  -m MOMENTUM, --momentum MOMENTUM
                        momentum (default: 0)
  -s SEED, --seed SEED  seed (default: 1)
```

then start a training run

```shell script
python train.py -bs=16
```

which should print the following to stdout

```less
Params(
  optimizer=SGD,
  learning_rate=0.01,
  dataset=MNIST,
  batch_size=16,
  epochs=10,
  momentum=0,
  seed=1
)
Starting training

name: MNIST-Stack
root: train-runs/MNIST-Stack/19-09-25T18:02:52
batch_size: 16
device: cpu

MODEL
=====

Stack(
  (infer0): Infer(
    (module): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  )
  (max_pool2d0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (re_lu0): ReLU(inplace=True)
  (infer1): Infer(
    (module): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  )
  (max_pool2d1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (re_lu1): ReLU(inplace=True)
  (flatten0): Flatten()
  (infer2): Infer(
    (module): Linear(in_features=320, out_features=50, bias=True)
  )
  (re_lu2): ReLU(inplace=True)
  (infer3): Infer(
    (module): Linear(in_features=50, out_features=10, bias=True)
  )
  (activation): LogSoftmax()
)


DATASET
=======

TransformDataset(
Dataset: Dataset MNIST
    Number of datapoints: 60000
    Root location: /Users/michal/.torch/datasets/MNIST
    Split: Train
Transforms: (Compose(
    ToTensor()
    Normalize(mean=(0.1307,), std=(0.3081,))
),)
)


LOADER
======

<torch.utils.data.dataloader.DataLoader object at 0x1a45cc8940>

LOSS
====

<function nll_loss at 0x120b700d0>


OPTIMIZER
=========

SGD (
Parameter Group 0
    dampening: 0
    lr: 0.01
    momentum: 0
    nesterov: False
    weight_decay: 0.0001
)

SCHEDULER
=========

None


PROGRESS
========
epochs: 0
steps: 0
samples: 0


Starting epoch 0

OPTIMIZER
=========

SGD (
Parameter Group 0
    dampening: 0
    lr: 0.01
    momentum: 0
    nesterov: False
    weight_decay: 0.0001
)


PROGRESS
========
epochs: 0
steps: 0
samples: 0


Batch inputs shape: (16, 1, 28, 28)
Batch targets shape: (16,)
Batch outputs shape: (16, 10)

batch:        0	accuracy: 0.1875	loss: 2.3783
batch:      128	accuracy: 0.6250	loss: 2.0528
batch:      256	accuracy: 0.6875	loss: 0.6222
```