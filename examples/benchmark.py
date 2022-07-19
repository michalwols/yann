import timm
import torch
from itertools import repeat
from torch.cuda.amp import autocast, GradScaler
import yann
from yann.callbacks import ProgressBar
import yann.transforms


class Params(yann.params.HyperParams):
  dataset = 'CIFAR10'

  size = 64
  model = 'resnet18'
  loss = 'cross_entropy'

  optimizer = 'SGD'

  batch_size = 32
  num_workers = 8
  pin_memory = False
  prefetch_factor = 2


  device = yann.default.device
  memory_format: str = 'contiguous_format'
  dtype = 'float32'
  non_blocking = True

  amp = False
  aot_autograd = False
  benchmark = False
  jit = False
  simple = False
  skip_loader = False


def simple_train_loop(
    model: torch.nn.Module,
    loader,
    loss,
    optimizer,
    device=None,
    memory_format=None,
    dtype=None,
    non_blocking=False,
    amp=False,
    progress: ProgressBar = None
):
  model.train()
  progress.on_epoch_start()
  for i, (inputs, targets) in enumerate(loader):

    # print(i)
    if device or memory_format:
      inputs, targets = (
        inputs.to(
          device,
          memory_format=memory_format,
          dtype=dtype,
          non_blocking=non_blocking
        ),
        targets.to(
          device=device,
          non_blocking=non_blocking
        )
      )

    with autocast(enabled=amp):
      outputs = model(inputs)
      l = loss(outputs, targets)

    optimizer.zero_grad(set_to_none=True)
    l.backward()
    optimizer.step()

    progress.on_step_end(inputs=inputs)
  progress.on_epoch_end()



def benchmark_train(params: Params):
  print(params)

  if params.amp:
    params.dtype = None

  memory_format = getattr(torch, params.memory_format)
  dtype = params.dtype and getattr(torch, params.dtype)

  model = timm.create_model(params.model)
  transform = yann.transforms.ImageTransformer(
    resize=params.size,
    crop=params.size,
    color_space='RGB'
  )

  dataset = yann.resolve.dataset(params.dataset, download=True)
  model.reset_classifier(len(dataset.classes))

  model = model.to(params.device)
  model = model.to(memory_format=memory_format)
  model = model.to(dtype=dtype)

  if params.jit:
    model = torch.jit.script(model)

  if params.aot_autograd:
    try:
      from functorch.compile import memory_efficient_fusion
    except ImportError:
      raise ValueError('functorch must be installed for aot_autograd support')
    model = memory_efficient_fusion(model)

  loader = yann.loader(
    dataset,
    transform=transform,
    batch_size=params.batch_size,
    num_workers=params.num_workers,
    pin_memory=params.pin_memory,
    prefetch_factor=params.prefetch_factor
  )

  if params.skip_loader:
    x, y = next(iter(loader))
    x = x.to(device=params.device, dtype=dtype, memory_format=memory_format)
    y = y.to(params.device)
    loader = repeat((x, y), 300)

  if params.benchmark:
    yann.benchmark()

  if params.simple:
    simple_train_loop(
      model=model,
      loader=loader,
      loss=yann.resolve.loss(params.loss),
      optimizer=yann.resolve.optimizer(params.optimizer, model.parameters()),
      device=params.device,
      dtype=dtype,
      memory_format=memory_format,
      non_blocking=params.non_blocking,
      amp=params.amp,
      progress=ProgressBar(length=len(dataset))
    )
  else:
    train = yann.train.Trainer(
      model=model,
      loader=loader,
      loss=yann.resolve.loss(params.loss),
      optimizer=yann.resolve.optimizer(params.optimizer, model.parameters()),
      device=params.device,
      # dtype=dtype,
      # memory_format=memory_format,
      amp=params.amp,
      callbacks=[ProgressBar(length=len(dataset))]
    )
    train(1)


if __name__ == '__main__':
  params = Params.from_command()
  benchmark_train(params)

