from yann.data.io.batch_files import BatchWriter
import torch
import numpy as np
import pytest


def test_batch_write_kwargs():
  with BatchWriter('asfasd') as write:
    for n in range(20):
      write.batch(
        ids=list(range(10)),
        targets=np.random.randn(10),
        outputs=torch.rand(10, 12),
      )


def test_batch_write_args():
  with BatchWriter('asfasd', names=('id', 'target', 'output')) as write:
    for n in range(20):
      write.batch(
        list(range(10)),
        np.random.randn(10),
        torch.rand(10, 12),
      )

  with pytest.raises(ValueError, 'names and encoders must be same length'):
    bw = BatchWriter('asfsd', names=(1,2,3), encoders=(1,2))


def test_meta():
  BatchWriter(path=lambda x: 'foo', meta={
    'checkpoint_id': 'asfads',
    'dataset': 'MNIST'
  })