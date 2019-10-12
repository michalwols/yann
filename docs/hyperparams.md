# Hyper Parameters

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

## Fields

## Validation

```python
params.validate()
```

## Watching for changes

```python
params.on_change(lambda k, v: print(f"changing {k} to {v}"))

params.learning_rate = 5
# > changing learning_rate to 5
```

## Sampling and Parameter Grids

```python
Params.sample()


Params.grid()
```

## Saving

```python
params.save('params.yml')
params.save('params.json')
```

## Loading

```python
params.load('params.yml')
params.load('params.json')
```


## Function Binding

```python

class Params(HyperParams):
  model = 'resnet50'

@params.bind()
def train(model, batch_size=32, optimizer='Adam'):
  pass


# train using the default parameters
train()

# override the parameters and update the params
train(model='seresnext50')
```