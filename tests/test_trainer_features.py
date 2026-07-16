"""Tests for gradient accumulation, token counting, and step signatures."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from yann.train import Trainer


def make_dataset(samples=20, features=10):
  return TensorDataset(
    torch.randn(samples, features),
    torch.randint(0, 2, (samples,)),
  )


def make_trainer(tmp_path, **kwargs):
  return Trainer(
    dataset=make_dataset(),
    batch_size=4,
    optimizer='SGD',
    lr=0.1,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=False,
    callbacks=False,
    root=tmp_path,
    **kwargs,
  )


def test_gradient_accumulation(tmp_path):
  # 20 samples, batch 4 -> 5 steps; accum 2 -> 2 full windows + trailing flush
  trainer = make_trainer(
    tmp_path,
    model=nn.Linear(10, 2),
    loss=nn.CrossEntropyLoss(),
    grad_accum=2,
  )
  before = trainer.model.weight.detach().clone()
  trainer.run(epochs=1)

  assert trainer.num_steps == 5
  assert trainer.num_optim_steps == 3
  assert trainer.num_samples == 20
  assert not torch.equal(before, trainer.model.weight)


class DictModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(10, 2)

  def forward(self, input_ids=None, attention_mask=None, labels=None):
    return self.lin(input_ids.float())


def dict_collate(items):
  inputs = torch.stack([x for x, _ in items])
  return {
    'input_ids': (inputs * 10).long(),
    'attention_mask': torch.ones(len(items), 10),
    'labels': torch.stack([y for _, y in items]),
  }


def test_raw_batch_step_and_token_counting(tmp_path):
  def step(trainer, batch):
    outputs = trainer.model(**batch)
    loss = nn.functional.cross_entropy(outputs, batch['labels'])
    trainer.update(loss=loss)
    return {'loss': loss.detach(), 'output': outputs}

  trainer = make_trainer(
    tmp_path,
    model=DictModel(),
    collate=dict_collate,
    step=step,
  )
  trainer.run(epochs=1)

  assert trainer.num_steps == 5
  assert trainer.num_samples == 20
  assert trainer.num_tokens == 200


def test_legacy_step_signature(tmp_path):
  def step(trainer, inputs, targets):
    outputs = trainer.model(inputs)
    loss = trainer.loss(outputs, targets)
    trainer.update(loss=loss)
    return outputs, loss

  trainer = make_trainer(
    tmp_path,
    model=nn.Linear(10, 2),
    loss=nn.CrossEntropyLoss(),
    step=step,
  )
  trainer.run(epochs=1)
  assert trainer.num_steps == 5
