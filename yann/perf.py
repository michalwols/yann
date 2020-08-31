def optimize(
  model=None,
  benchmark=True,
  jit=True,
  amp=True,
  xla=False,
  quantize=False,
  freeze=False,
  example=None,
  device=None
):
  if benchmark:
    from torch.backends import cudnn
    cudnn.benchmark = True

  if model:
    if example:
      if isinstance(example, tuple):
        import torch
        example = torch.randn(*example)

      if jit:
        import torch.jit
        model = torch.jit.trace(model, example)
  return model

