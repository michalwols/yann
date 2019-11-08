from pyarrow import parquet as pq
import pyarrow as pa
import pandas as pd


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