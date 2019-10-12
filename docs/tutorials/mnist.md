# LeNet in 30 Seconds

### Hyper Parameters to command line

Step one of any good experiment is defining the parameters

```python
import yann
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
```

This will automatically generate a command line interface for your experiment, making it easy to try different configurations

```commandline
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


next we'll set the seed to make our experiment reproducible
```python
# set random, numpy and pytorch seeds in one call
yann.seed(params.seed)
```

### Model Definition with Shape Inference

Calculating the input shape of your layers can be a pain, with the `Infer()` module we can let the framework do it for us.

```python
from torch import nn
from yann.modules import Stack, Infer

lenet = Stack(
  Infer(nn.Conv2d, 10, kernel_size=5),
  nn.MaxPool2d(2),
  nn.ReLU(),

  Infer(nn.Conv2d, 20, kernel_size=5),
  nn.MaxPool2d(2),
  nn.ReLU(),

  nn.Flatten(),

  Infer(nn.Linear, 50),
  nn.ReLU(),
  Infer(nn.Linear, 10),

  activation=nn.LogSoftmax(dim=1)
)

# run a forward pass to infer input shapes using `Infer` modules
lenet(torch.rand(1, 1, 28, 28))
```


### Configure a Trainer


```python
import torch
from torch import nn
from torchvision import transforms

import yann
from yann.train import Trainer


# use the registry to resolve optimizer name to an optimizer class
optimizer = yann.resolve.optimizer(
  params.optimizer,
  yann.trainable(lenet.parameters()),
  momentum=params.momentum,
  lr=params.learning_rate
)

train = Trainer(
  model=lenet,
  optimizer=params.optimizer,
  dataset=params.dataset,
  batch_size=params.batch_size,
  transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ]),
  loss='nll_loss',
  metrics=('accuracy', 'top_3_accuracy'),
)

train(params.epochs)

# save checkpoint
train.checkpoint()

# plot the loss curve
train.history.plot()
```