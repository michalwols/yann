import pathlib

from yann.utils import print_tree


class Paths:
  def __init__(self, root):
    self.root = pathlib.Path(root)

    self.checkpoint_format = '{}{}'

  def create(self):
    self.root.mkdir(parents=True, exist_ok=True)

  @property
  def checkpoints(self):
    path = self.root / 'checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path

  def checkpoint(self, **kwargs):
    return self.checkpoints / self.checkpoint_format.format(**kwargs)

  @property
  def tensorboard(self):
    return self.root / 'tensorboard'

  @property
  def logs(self):
    return self.root / 'logs'

  @property
  def evals(self):
    return self.root / 'evals'

  @property
  def plots(self):
    return self.root / 'plots'

  @property
  def outputs(self):
    return self.root / 'outputs'

  @property
  def exports(self):
    return self.root / 'exports'

  @property
  def summary(self):
    return self.root / 'summary.yaml'

  @property
  def profile(self):
    return self.root / 'profile'

  @property
  def git_diff(self):
    return self.root / 'git.diff'

  @property
  def requirements(self):
    return self.root / 'requirements.txt'

  def tree(self, **kwargs):
    print_tree(self.root, **kwargs)
