
from torch.nn.functional import relu, max_pool2d
from yann.layers import Conv, Pool, Dropout, Softmax, Stack
from yann import BaseTrainer


def LeNet(activation=relu, pool=max_pool2d):
  return Stack(
    Conv(2, 20, shape=5, pad='same'),
    activation,
    pool,
    Conv(2, 20, (5, 5), pad='same'),
  )


model = LeNet()


train = BaseTrainer(model, 'mse', device='gpu', data='cifar10')

train(epochs=30, batch_size=20)

train.save()


####


train = BaseTrainer.load(model='lenet', loss='mse', checkpoint='latest')

