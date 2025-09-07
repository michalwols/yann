def zero_out_nans(module, grad_input, grad_output):
  """
  Backwards hook to turn NaNs in gradients to 0
  """
  for grad in grad_input:
    grad[grad != grad] = 0  # technically shouldn't modify inputs


class Hook:
  def __init__(self, module):
    self.module = module
    self.handles = None
    self.register(self.module)

  def register(self, module):
    pass

  def remove(self):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.remove()


def shape_hook(name, depth=None, show=False):
  depth = depth if depth is not None else name.count('.')
  indent = '  ⬐' * depth if name else ''

  def print_shapes(m, input, output):
    print(
      f'{f"{indent}{name}:": <20}',
      f'{f"({m.__class__.__name__})": <15}',
      ', '.join(f'{str(tuple(x.shape)): <15}' for x in input),
      '=>',
      f'{str(tuple(output.shape)): <15}',
    )

    if show is not False:
      import yann

      if show is True:
        yann.show(output)
      else:
        yann.show(output[show])

  return print_shapes


class ShapeHook(Hook):
  def __init__(self, module, show=False):
    self.show = show
    super().__init__(module)

  def register(self, module):
    self.handles = []
    for n, m in module.named_modules():
      self.handles.append(
        m.register_forward_hook(shape_hook(n, show=self.show)),
      )

  def remove(self):
    for h in self.handles:
      h.remove()
