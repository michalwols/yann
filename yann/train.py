
from layers.core import Function, Layer


class BaseTrainer(Function):
  def __init__(self, *args, **kwargs):
    # self.function = func
    # self.loss = loss

    # self.data = data

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


class Trainer(BaseTrainer):
  model: Layer

  def __init__(self, model, loss, optimizer, device='gpu', data=None,
               batch_size='fit', epochs=None):
    super(Trainer, self).__init__()
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
    self.data = data

  def step(self, inputs, target):
    self.model.is_training = True
    self.model.grads.zero()

    outputs = self.model(inputs)

    loss = self.loss(outputs, target)
    loss.backward()

    self.optimizer.step()
    self.num_steps += 1

    return outputs, loss


class SiameseTrainer(BaseTrainer):
  pass


class TripletTrainer(BaseTrainer):
  pass


class CTCTrainer(BaseTrainer):
  pass


class TeacherTrainer(BaseTrainer):
  def __init__(self, student, teacher):
    super(TeacherTrainer, self).__init__()

    self.student = student
    self.teacher = teacher

  def step(self):
    pass
