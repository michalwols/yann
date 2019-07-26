import pytest
import torch.cuda
from torch import nn
from torch.optim import SGD

from yann.callbacks import (
  History, HistoryPlotter, HistoryWriter, Logger,
  Checkpoint
)
from yann.data.datasets import TinyDigits
from yann.modules import Flatten
from yann.train import Trainer

devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


@pytest.mark.slow
@pytest.mark.parametrize('device', devices)
def test_train(tmpdir, device):
  """Sanity check train run"""

  model = nn.Sequential(
    nn.Conv2d(1, 20, 3),
    nn.ReLU(inplace=True),
    nn.Conv2d(20, 20, 3),
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(320, 10)
  )

  train = Trainer(
    root=tmpdir,
    model=model,
    dataset=TinyDigits(),
    device=device,
    optimizer=SGD(
      model.parameters(),
      lr=.01, momentum=0.9, weight_decay=.001
    ),
    loss=nn.CrossEntropyLoss(),
    callbacks=[
      History(),
      HistoryPlotter(save=True),
      HistoryWriter(),
      Logger(batch_freq=20),
      Checkpoint()
    ]
  )

  train(10)

  assert train.checkpoints_root.is_dir()
  assert train.history.metrics

  export_path = train.export()

  assert export_path
  assert export_path.is_dir()


@pytest.mark.slow
@pytest.mark.parametrize('device', devices)
def test_train_resolved(tmpdir, device):
  from yann.data.transform import ImageTransformer
  
  train = Trainer(
    root=tmpdir,
    model='densenet121',
    dataset='CIFAR10',
    loss='CrossEntropy',
    optimizer='SGD',
    transform=ImageTransformer(resize=224)
  )

  train(1)
