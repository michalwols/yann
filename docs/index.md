# Introduction

Yann is a batteries included deep learning framework built in PyTorch.

Inspired by Django and Rails, it aims to automate the tedious steps of a machine learning project 
so that you can focus on the (fun) hard parts. It makes it easy to quickly get a project started 
and scales with you all the way to production.
 
It could also be viewed as `#!python torch.nn` extended, as it includes common new research modules 
that might be too experimental to be included in torch.


## Getting Started

### Install 

```commandline
pip install yann
```


## Quick Tour

### Flexible Trainer

Yann provides a `Trainer` class that encapsulates your experiment state and handles common tasks 
such as object instantiation, checkpointing and progress tracking. 

#### Default Training Loop

```python
from yann.train import Trainer
from yann.transforms import Compose, ToTensor, Resize, Normalize

train = Trainer(
  model='resnet50',  # could also be an instance 
  dataset='Imagenette160',
  transform=Compose([
    Resize(224),
    Normalize('imagenet'),
    ToTensor()
  ]),
  batch_size=32,
  optimizer='AdamW',
  loss='cross_entropy',
  metrics=('accuracy', 'top_3_accuracy'),
  device='cuda:0'
)

# run training for 5 epochs
train(epochs=5)

# save checkpoint, including model and optimizer state
train.checkpoint(name='{time}-{loss}-{steps}.th')

train.history.plot()
```


#### Custom Logic

It exposes methods for common tasks such as iterating over data batches and checkpointing, making it a convenient state container for more complicated uses cases.  

As an example we can implement [Accelerating Deep Learning by Focusing on the Biggest Losers](https://arxiv.org/abs/1910.00762) using an inverted loop
```python
train.checkpoint.load('latest')

for epoch in train.epochs(5):  # keeps track of epochs, and starts iteration from current epoch (even after loading a checkpoint)
  for inputs, targets in train.batches():  # yield training batches from the train data loader
    with yann.optim_step(train.optimizer):  # calls optimizer.zero_grad() and optimizer.step() for you

      with torch.no_grad():
       outputs = train.model(inputs)
       losses = train.loss(outputs, targets, reduction='none')
    
      _, top_indices = losses.topk(12)
    
      outputs = train.model(inputs[top_indices])
      loss = train.loss(outputs, targets[top_indices])
      loss.backward()

  train.checkpoint()
```

which could also be done by passing a step function when initializing the trainer (`#!python Trainer(step=step_on_top_losses)`) or by using the `override` decorator

```python
@train.override('step')
def step_on_top_losses(train: Trainer, inputs, targets):
  train.model.train()
  train.optimizer.zero_grad()
    
  with torch.no_grad():
    outputs = train.model(inputs)
    losses = train.loss(outputs, targets, reduce='none')
    
  _, top_indices = losses.topk(12)
    
  outputs = train.model(inputs[top_indices])
  loss = train.loss(outputs, targets[top_indices])
  loss.backward()
  train.optimizer.step()
```

#### Callbacks

Inspired by Keras, the trainer supports functional or class based callbacks that make it easy to integrate additional event handlers during the training process. 

##### Function Based
```python

@train.on('batch_end')
def plot_scores(inputs, targets, outputs, loss, **kwargs):
  yann.plot.scores(outputs)

@train.on('batch_error')
def handle_error(error):
  ...
```

##### Class Based
```python
from yann.callbacks import HistoryPlotter

train = Trainer(callbacks=(HistoryPlotter(),))

# or add it later
train.callbacks.append(HistoryPlotter())


```


#### Experiment Tracking and Reproducibility

To help you track your experiments and keep things reproducible, the trainer automatically tracks your git hash, python dependencies, logs and checkpoints in `train.paths.root`.

### Hyperparamter Definition

```python
from yann.params import HyperParams, Choice, Range


class Params(HyperParams):
  dataset = 'MNIST'
  batch_size = 32
  epochs = 10
  optimizer: Choice(('SGD', 'Adam')) = 'SGD'
  learning_rate: Range(.01, .0001) = .01
  momentum = 0

  seed = 1
```

#### Automatic Command Line Interface



```python
# parse command line arguments
params = Params.from_command()
```

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
