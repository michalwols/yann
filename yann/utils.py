import subprocess


def gcp_sync(src, dst):
  subprocess.call([
    'gsutil',
    '-m',
    'rsync',
    '-r',
    src,
    dst
  ])
