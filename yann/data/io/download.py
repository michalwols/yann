import urllib.request
from concurrent import futures
import pathlib
from pathlib import Path
from urllib.parse import urlparse
import os

from ...utils import progress


class CachedExecutor:
  def __init__(self, workers=8):
    self._executor = futures.ThreadPoolExecutor(max_workers=workers)

    self.pending = {}  # key => future
    self.results = {} # key => local path
    self.errors = {}  # key => error

    self.error_callbacks = []
    self.success_callbacks = []
    self.cancel_callbacks = []

  def on_error(self, callback):
    self.error_callbacks.append(callback)

  def on_success(self, callback):
    self.success_callbacks.append(callback)

  def on_cancel(self, callback):
    self.cancel_callbacks.append(callback)

  def __del__(self):
    self.cancel()
    if self._executor:
      self._executor.shutdown()

  def cancel(self):
    for key, future in self.pending.items():
      future.cancel()

  def complete(self):
    self._executor.shutdown()

  def as_completed(self):
    yield from futures.as_completed(self.pending.values())

  def wait(self, *args, **kwargs):
    return futures.wait(self.pending.values(), *args, **kwargs)

  def handle_done_future(self, future: futures.Future):
    self.pending.pop(future.key)

    if future.cancelled():
      for c in self.cancel_callbacks:
        c(future.key, future)
      return

    try:
      r = future.result()
      self.results[future.key] = r
    except Exception as e:
      self.errors[future.key] = e
      for c in self.error_callbacks:
        c(future.key, e, future)
      return

    for c in self.success_callbacks:
      c(future.key, r, future)

  def execute(self, key, *args, **kwargs):
    raise NotImplementedError()

  def submit(self, key, *args, **kwargs):
    future = self._executor.submit(self.execute, key, *args, **kwargs)
    self.pending[key] = future
    future.key = key
    future.add_done_callback(self.handle_done_future)
    return future

  def prefetch(self, keys):
    futures = []
    for k in keys:
      f = self.enqueue(k)
      if f:
        futures.append(f)
    return futures

  def enqueue(self, key, *args, **kwargs):
    if key in self.results or key in self.pending:
      return
    return self.submit(key,  *args, **kwargs)

  def get(self, key):
    if key in self.results:
      return self.results[key]
    if key in self.pending:
      return self.pending[key].result()

    return self.submit(key).result()

  def __getitem__(self, key):
    return self.get(key)

  def __contains__(self,key):
    return (
      key in self.results
      or key in self.errors
      or key in self.pending
    )

  def __delitem__(self, key):
    self.results.pop(key)
    self.errors.pop(key)
    p = self.pending.pop(key)
    if p:
      p.cancel()

class Downloader(CachedExecutor):
  def __init__(self, local_root='./', workers=8):
    super(Downloader, self).__init__(workers=workers)
    self.local_root = local_root

  def execute(self, key, *args, **kwargs):
    uri = self.get_uri(key)
    dest = self.get_path(uri)
    return self.download(uri, dest)[0]

  def get_path(self, url):
    return None

  def get_uri(self, key):
    return key

  def download(self, uri, path):
    return download(uri, path, root=self.local_root)


def download(url, dest=None, skip_existing=True, nest=True, root='./'):
  """
  Returns: (local_path, headers), headers will be None if file exists
  """
  if not dest:
    root = os.path.abspath(root)
    dest = (urlparse(url).path if nest else os.path.basename(urlparse(url).path))
    dest = os.path.join(root, dest[1:] if dest[0] == '/' else dest)
  elif hasattr(dest, '__call__'):
    dest = dest(url)

  os.makedirs(os.path.dirname(dest), exist_ok=True)

  if skip_existing and pathlib.Path(dest).exists():
    return (dest, None)
  return urllib.request.urlretrieve(url, dest)


def download_urls(urls, dest=None, skip_existing=True, nest=True, root='./',
                  max_workers=12):
  results, errors = [], []
  with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    queued_futures = {
      executor.submit(
        download,
        url,
        dest=dest,
        skip_existing=skip_existing,
        nest=nest,
        root=root
      ): url
      for url in urls
    }
    for f in progress(futures.as_completed(queued_futures)):
      try:
        results.append(f.result())
      except Exception as e:
        errors.append((queued_futures[f], e, f))
  return results, errors