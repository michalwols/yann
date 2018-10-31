import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.optim import SGD
from torch.utils.data import TensorDataset
from torchvision.transforms import ToTensor

from yann.callbacks import (
  History, HistoryPlotter, HistoryWriter, Logger,
  Checkpoint
)
from yann.layers import Flatten
from yann.train import Trainer


def test_train(tmpdir):
  """Sanity check train run"""
  h = History()

  digits = load_digits()

  t = ToTensor()

  dataset = TensorDataset(
    torch.from_numpy(digits.images).unsqueeze(1).float(),
    torch.Tensor(digits.target).long()
  )

  model = nn.Sequential(
    nn.Conv2d(1, 10, 3),
    nn.ReLU(inplace=True),
    nn.Conv2d(10, 3, 3),
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(48, 10)
  )

  train = Trainer(
    root=tmpdir,
    model=model,
    dataset=dataset,
    optimizer=SGD(
      model.parameters(),
      lr=.01, momentum=0.9, weight_decay=.001),
    loss=nn.CrossEntropyLoss(),
    callbacks=[
      h,
      HistoryPlotter(h, save=True),
      HistoryWriter(),
      Logger(),
      Checkpoint()
    ]
  )

  train(10)
  train.export(tmpdir / 'export')
