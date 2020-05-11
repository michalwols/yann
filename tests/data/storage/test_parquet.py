from yann.data.storage.parquet import BatchParquetFileWriter

import torch



def test_parquet_batch_writer(tmpdir):
  path = tmpdir / 'test.parquet'
  with BatchParquetFileWriter(path) as write:
    for i in range(10):
      write.batch(
        ids=list(range(10)),
        labels=torch.ones(10, 12)
      )

  assert path.exists()