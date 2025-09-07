import shutil
import subprocess


def run(command):
  out = subprocess.check_output(command, shell=True)
  if out:
    return out.decode('utf-8').strip()


def git_hash():
  return run('git rev-parse --short HEAD')


def git_commit(files='.', message='automated commit', branch=None):
  if branch:
    try:
      run(f'git checkout {branch}')
    except subprocess.SubprocessError:
      run(f'git checkout -b {branch}')
  if isinstance(files, str):
    run(f'git add {files}')
  else:
    run(['git', 'add', *files])
  run(f'git commit -m {message}')


def git_diff():
  return run('git diff')


def shutdown_computer():
  return run('sudo shutdown -h now')


def nvidia_smi():
  return run('nvidia-smi')


def pip_freeze():
  try:
    # Try standard pip first
    return run('pip freeze')
  except subprocess.CalledProcessError as e:
    if e.returncode == 127:
      # If pip is not found, try uv pip freeze
      try:
        return run('uv pip freeze')
      except subprocess.CalledProcessError as uv_e:
        print(
          f"Warning: 'pip freeze' and 'uv pip freeze' failed. Requirements will not be saved. Error: {uv_e}",
        )
        return 'Could not determine requirements.'
    else:
      # Re-raise other pip errors
      raise e


def conda_list(explicit=False):
  return run(f'conda list {"--explicit" if explicit else ""}')
