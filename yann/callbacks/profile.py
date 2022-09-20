from . import Callback


class Profile(Callback):
  def __init__(
      self,
      start_step=16,
      stop_step=80,
      **kwargs
  ):
    self.profiler = None

    self.start_step = start_step
    self.stop_step = stop_step
    # self.tensorboard = tensorboard

    self.profiler_args = kwargs


  def on_step_start(self, index=None, **kwargs):
    if index == self.start_step:
      from torch.profiler import profile
      self.profiler = profile(**kwargs)
      self.profiler.start()

  def on_step_end(self, index=None, trainer=None, **kwargs):
    if self.start_step <= index <= self.stop_step:
      self.profiler.step()
    if index == self.stop_step:
      self.profiler.stop()
      self.save(root=trainer.paths.profile)
      self.disable()


  def save(self, root: 'pathlib.Path' = None):
     self.profiler.export_chrome_trace(str(root / 'chrome_trace.json'))

     try:
      self.profiler.tensorboard_trace_handler(str(root / 'tensorboard'))
     except:
       pass

     try:
       self.profiler.export_stacks(
         str(root / 'cpu.stacks'),
         metric='self_cpu_time_total'
       )
     except:
       pass

     try:
       self.profiler.export_stacks(
         str(root / 'cuda.stacks'),
         metric='self_cuda_time_total'
       )
     except:
       pass
