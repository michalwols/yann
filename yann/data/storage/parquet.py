from pyarrow import parquet as pq
import pyarrow as pa
import pandas as pd

from .batch_files import BatchWriter

def write_parquet(dest, data, columns=None, **kwargs):
  if isinstance(data, dict):
    data = pd.DataFrame(data)

  if isinstance(data, pd.DataFrame):
    return data.to_parquet(dest, **kwargs)
  else:
    d = next(data)
    df = pd.DataFrame(d)
    table = pa.Table.from_pandas(df)
    with pq.ParquetWriter(dest, schema=table.schema, **kwargs) as writer:
      writer.write_table(table)
      for d in next(data):
        writer.write_table(pa.Table.from_pandas(pd.DataFrame(d))


def read_parquet():
  pass


class BatchParquetFileWriter(BatchWriter):
  def __init__(self, path, schema=None, encoders=None, names=None, meta=None, **writer_args):
    super(BatchParquetFileWriter, self).__init__(
      path=path,
      batches_per_file=1,  # writing to a parquet
      encoders=encoders,
      names=names,
      meta=meta
    )

    self.path = path
    self.schema = schema
    self._writer_args = writer_args

    # will determine schema on first write if not provided
    if self.schema:
      self.writer= pq.ParquetWriter(path, self.schema, **self._writer_args)

  def _write(self):
    df = pd.DataFrame(self.collate(self.buffers))
    table = pa.Table.from_pandas(df)
    if self.writer is None:
      self.schema = table.schema
      self.writer = pq.ParquetWriter(self.path, self.schema, **self._writer_args)
    self.writer.write_table(table)


class BatchParquetDatasetWriter(BatchWriter):
  pass