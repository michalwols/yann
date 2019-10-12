```python
from yann.train import Trainer

train = Trainer(
  model='resnet18',
  dataset='MNIST',
  optimizer='AdamW',
  loss='cross_entropy'
)
```


### Register callbacks

```python
@train.on('epoch_end')
def sync_data():
  pass
```

### Implement custom step logic

```python

for e in train.epochs(4):
  for inputs, targets in train.batches():
    train.optimizer.zero_grad()
    outputs = train.model(inputs)
    loss = train.loss(outputs, targets)
    loss.backwards()

  train.checkpoint()
```

## Checkpointing

```python
train.checkpoint()
```

### Continue where you left off

```python
train.load_checkpoint('latest')
```


## History

```python
train.history.plot()
```



---

## Functional Interface

```python
from yann.train import train

for epoch in range(10):
  for _ in train(model, loader, optimizer, loss='cross_entropy', device='cuda'):
    pass
```