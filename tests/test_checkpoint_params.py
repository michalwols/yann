"""Checkpoints record the config they were trained with."""

import hp
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from yann.train import Trainer


def make_trainer(tmp_path, **kwargs):
  kwargs.setdefault('callbacks', False)
  return Trainer(
    model=nn.Linear(10, 2),
    dataset=TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))),
    batch_size=4,
    optimizer='SGD',
    loss=nn.CrossEntropyLoss(),
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=False,
    root=tmp_path,
    **kwargs,
  )


def test_checkpoint_records_the_config(tmp_path):
  trainer = make_trainer(tmp_path, lr=0.05, epochs=3)
  trainer(epochs=1)
  path = trainer.checkpoint()

  saved = torch.load(path, weights_only=False)
  assert saved['params']['lr'] == 0.05
  assert saved['params']['epochs'] == 3
  assert saved['params']['batch_size'] == 4


def test_recorded_config_is_serializable(tmp_path):
  import json

  trainer = make_trainer(tmp_path, lr=0.05)
  saved = trainer.state_dict()['params']

  # the model and dataset become type names rather than pickled objects
  assert isinstance(saved['model'], str)
  assert isinstance(saved['dataset'], str)
  json.dumps(saved)


def test_secrets_never_reach_the_checkpoint(tmp_path):
  class Secretive(Trainer.Params):
    token: str = hp.Field(default='sk-live', secret=True)

  trainer = make_trainer(tmp_path, params=Secretive(lr=0.05))
  assert 'token' not in trainer.state_dict()['params']


def test_loading_keeps_the_checkpoint_config(tmp_path):
  trainer = make_trainer(tmp_path, lr=0.05)
  path = trainer.checkpoint()

  other = make_trainer(tmp_path, lr=0.05)
  other.load_checkpoint(path)
  assert other.checkpoint_params['lr'] == 0.05


def test_loading_warns_when_the_config_disagrees(tmp_path, caplog):
  trainer = make_trainer(tmp_path, lr=0.05)
  path = trainer.checkpoint()

  mismatched = make_trainer(tmp_path, lr=0.5)
  with caplog.at_level('WARNING'):
    mismatched.load_checkpoint(path)

  assert any('trained with different params' in r.message for r in caplog.records)


def test_from_checkpoint_rebuilds_with_the_saved_config(tmp_path):
  trainer = make_trainer(tmp_path, lr=0.05, epochs=7)
  trainer(epochs=1)
  path = trainer.checkpoint()

  resumed = Trainer.from_checkpoint(
    path,
    model=nn.Linear(10, 2),
    loss=nn.CrossEntropyLoss(),
    dataset=TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))),
    root=tmp_path,
    callbacks=False,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=False,
  )

  assert resumed.params.lr == 0.05          # from the checkpoint
  assert resumed.params.epochs == 7
  assert resumed.num_steps == trainer.num_steps   # counters restored


def test_from_checkpoint_overrides_win(tmp_path):
  trainer = make_trainer(tmp_path, lr=0.05)
  path = trainer.checkpoint()

  resumed = Trainer.from_checkpoint(
    path,
    lr=0.001,
    model=nn.Linear(10, 2),
    loss=nn.CrossEntropyLoss(),
    dataset=TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))),
    root=tmp_path,
    callbacks=False,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=False,
  )
  assert resumed.params.lr == 0.001
