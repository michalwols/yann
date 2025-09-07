from functools import cached_property
from typing import Iterable

import duckdb


class DuckDBDataset:
  def __init__(
    self,
    query: str,
    connection: duckdb.DuckDBPyConnection = None,
    use_rowid: bool = True,
  ):
    self.query = query
    self.connection = connection or duckdb.connect()

    self._use_rowid = use_rowid

  @cached_property
  def row_count(self):
    return self.connection.execute(
      f'SELECT COUNT(*) FROM ({self.query})',
    ).fetchone()[0]

  def __len__(self):
    return self.row_count

  def __getitem__(self, item):
    if isinstance(item, Iterable):
      return self.connection.execute(
        f'SELECT * FROM ({self.query}) WHERE rowid in ({",".join(map(str, item))})'
        if self._use_rowid
        else f"""
          SELECT * EXCLUDE (row_number) FROM (
            SELECT ROW_NUMBER() OVER () as row_number, 
            * 
            FROM ({self.query})
          )
          WHERE row_number - 1 in ({','.join(map(str, item))})
        """,
      ).arrow()
    else:
      return self.connection.execute(
        f'SELECT * FROM ({self.query}) WHERE rowid = {item}'
        if self._use_rowid
        else f"""
          SELECT * EXCLUDE (row_number) FROM (
            SELECT ROW_NUMBER() OVER () as row_number, 
            * 
            FROM ({self.query})
          )
          WHERE row_number - 1 = {item}
        """,
      ).arrow()


class DuckDBIterableDataset:
  def __init__(
    self,
    query: str,
    connection: duckdb.DuckDBPyConnection = None,
    batch_size: int = 1,
  ):
    self.query = query
    self.connection = connection or duckdb.connect()

    self.batch_size = batch_size

  def __iter__(self):
    self.cursor = self.connection.execute(self.query)

    while True:
      rows = self.cursor.arrow(rows_per_batch=self.batch_size)
      if not rows:
        break
      yield rows
