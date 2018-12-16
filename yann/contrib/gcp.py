import subprocess

from ..callbacks.base import Callback


def run(command):
  return subprocess.check_call(command, shell=True)


def gcloud(command):
  return run(['gcloud', command])


def gsutil(command):
  return run(['gsutil', command])


def args(*flags, hyphenate=True, **kwargs):
  return ' \ \n'.join((
    *(f'--{str(n).replace("_", "-") if hyphenate else n}' for n in flags),
    *(x for x in (
      f'--{str(k).replace("_", "-") if hyphenate else k}={v}'
      if not (v is True or v is False)
      else (f'--{str(k).replace("_", "-") if hyphenate else k}' if v else '')
      for k, v in kwargs.items() if v is not None)
      if x)
  ))


def start_instance(
    name,
    zone=None,
    preemptible=True):
  command = (
    f"""gcloud compute instances create {name} \
          {
    args(
      zone=zone,
      preemptible=preemptible,
      maintenance_policy='foo',
    )
    }

    """
  )

  return run(command)


def start_dl_instance(name, ):
  pass


def copy_directory(src, dst):
  pass


def stop_instance():
  pass


def kill_instance():
  pass


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
