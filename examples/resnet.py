from yann.layers import (
  Conv2D, Pool, Residual, Stack, Conv, Dense,
  MaxPool, Convolution2D, Flatten, Dropout, Softmax
)

from yann.train import Trainer


def Resnet():
  return Stack(
    Conv2D(64, 5),
    Pool('avg', 2)
  )


conv = Conv2D(64, 3)

conv.grads.zero()

conv.params.all()


resnet = Resnet()

convs = resnet.layers[Conv]
params = resnet.params.all()

resnet.grads.zero()

resnet.device('gpu')
