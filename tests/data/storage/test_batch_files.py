from yann.data.storage.batch_files import BatchWriter, PartitionedBatchWriter, BatchReader
import numpy as np
import torch
import  pathlib

import yann

def same_type(*items):
  return all(type(x) == type(items[0]) for x in items)


def test_pickle(tmpdir: pathlib.Path):
  path = tmpdir / 'batches.pkl'
  w = BatchWriter(path, names=('ids', 'targets', 'outputs', 'paths'))

  batches = []

  for i in range(10):
    batches.append((
      list(range(10)),
      torch.zeros(10, 12),
      torch.rand(10, 12),
      [f"{i}-{n}.jpg" for n in range(10)]
    ))

    w.batch(*batches[-1])

  w.close()

  assert path.exists()
  # assert path.stat().st_size > 400

  assert w.meta_path.exists()

  assert w.path == path

  loaded_batches = yann.load(w.path)

  assert len(loaded_batches) == 10




def test_use_case(tmpdir):
  model = torch.nn.Module()

  w = BatchWriter(tmpdir / 'MNIST-preds.pkl')

  iw = BatchWriter(tmpdir/'inputs.pkl')

  for inputs, targets in iw.through(yann.batches('MNIST', size=32, workers=10, transform=())):
    preds = model(inputs)
    w.batch(
      targets=targets,
      preds=preds
    )
  w.close()

  processed = 0
  correct = 0
  r = BatchReader(w.path)
  for batch in r.batches():
    processed += len(batch['targets'])
    correct += sum(batch['targets'] == batch['preds'])


