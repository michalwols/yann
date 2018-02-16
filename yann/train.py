
from layers.core import Function


class Trainer(Function):
  def __init__(self, func, *args, loss=None, data=None, **kwargs):
    self.function = func
    self.loss = loss

    self.data = data

    self.num_steps = 0

  def step(self, *inputs, **kwargs):
    """Run a single training step"""
    self.num_steps += 1

  def run(self, data, epochs=1, batch_size=None, **kwargs):
    pass

  def __call__(self, epochs=1, **kwargs):
    pass

  def device(self, name):
    pass

