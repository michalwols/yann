from yann.layers import (
  Stack, Dense,
  MaxPool, Convolution2D, Softmax
)

from yann.train import Trainer


model = Stack(
  Convolution2D(10, 5, activation='relu'),
  MaxPool(2),
  Convolution2D(29, 5, dropout=.4),
  # Flatten(),  # might make sense to roll this into dense
  Dense(activation='relu', dropout=.5),
  Dense(),
  Softmax(log=True)
)

train = Trainer(
  model=model,
  data='mnist',
  optimizer='sgd',
  loss='nll',
  device='gpu',
  batch_size='fit',
  epochs=100
)

train(epochs=20)