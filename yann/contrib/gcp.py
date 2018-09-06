import subprocess

from yann.callbacks.base import Callback

def gcp_sync(src, dst, exclude=None):
  if exclude:
    return subprocess.call([
      'gsutil',
      '-m',
      'rsync',
      '-r',
      '-x',
      exclude,
      src,
      dst
    ])
  else:
    return subprocess.call([
      'gsutil',
      '-m',
      'rsync',
      '-r',
      src,
      dst
    ])



class SyncCallback(Callback):
  def upload(self):
    pass

  def download(self):
    pass


class GCPSync(SyncCallback):
  pass