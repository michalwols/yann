from torch.autograd.profiler import profile
from typing import Union, Tuple, Callable
import torch
from .timer import Timer

# TODO:
# - https://github.com/pytorch/pytorch/issues/3749#issuecomment-374006211
# - https://pytorch.org/docs/stable/bottleneck.html

def param_count(model):
  return sum(p.numel() for p in model.parameters())


def profile_module(
    module,
    input,
    warmup=10,
    iterations=20,
    sync=True,
    timer=None,
    task_name='iteration',
    jit=False
):
  if jit:
    import torch.jit
    module = torch.jit.trace(module, input)

  for n in range(warmup):
    module(input)

  t = timer or Timer()
  tasks = []
  for n in range(iterations):
    with t.task(task_name, sync=sync) as task:
      module(input)
      tasks.append(task)

  return tasks
