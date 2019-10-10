from typing import Union, Callable, Iterable
import time
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.jit
import torch
import yann

from ..data.loaders import TransformLoader


def inference_stream(
    model: Union[nn.Module, Callable, str],
    data: Union[Dataset, DataLoader, Iterable, str],
    device=None,
    transform=None,
    batch_size=64,
    parallel=False,
    num_workers=1,
    pin_memory=False,
    shuffle=False,
    progress=10,
    eval=True,
):
  device = device or yann.default_device

  if isinstance(model, str):
    model = torch.jit.load(model, 'cpu')

  if isinstance(model, nn.Module):
    if parallel:
      model = nn.DataParallel(model)
    if eval: model.eval()
    model.to(device)

  if isinstance(data, str):
    data = yann.resolve.dataset(data)

  if isinstance(data, Dataset):
    data = TransformLoader(
      data,
      transforms=transform,
      pin_memory=pin_memory,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=num_workers
    )

  try:
    size = len(data)
  except:
    size = -1

  start = begin = time.time()

  with torch.no_grad():
    for idx, (inputs, *rest) in enumerate(data):
      inputs = inputs.to(device)
      outputs = model(inputs)

      yield (inputs, *rest, outputs)

      if progress and idx % progress == 0:
        print(f"[{idx} / {size}] ({time.time() - begin}, total: {time.time() - start})")

      begin = time.time()

