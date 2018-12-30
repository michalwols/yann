import urllib.request
from concurrent import futures
import pathlib
from urllib.parse import urlparse
import os

from ...utils import progress



def download(url, dest=None, skip_existing=True, nest=True, root='./'):
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