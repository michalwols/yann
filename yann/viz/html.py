from abc import ABCMeta

class styles(dict):
  def __init__(self, *args, **props):
    super().__init__()
    if args:
      if isinstance(args[0], dict):
        self.update(args[0])
    if isinstance(args[0], str):
      for l in args[0].split(';'):
        l = l.strip()
        if l:
          k, v = l.split(':')

          self[k.replace('-', '_').strip()] = v.strip()

    self.update(props)

  def __str__(self):
    return ' '.join(f"{k.replace('_', '-')}: {v};"
                    for k, v in self.items())


class Node:
  NAME = None
  CHILDREN = False

  def __init__(self, *children, style=None, **props):
    self._display_handle = None
    self.children = list(children)
    self.props = props
    self.style = style if isinstance(style, styles) else styles(style)

  @property
  def name(self):
    return self.NAME or self.__class__.__name__

  def __str__(self):
    return self.html()

  __repr__ = __str__

  _repr_html_ = __str__

  def html(self):
    if self.CHILDREN:
      return f"""
         <{self.name} {' '.join(
        f'{k}="{v}"' for k, v in self.props.items())}  style="{self.style}"/>
      """

    return f"""
    <{self.name} {' '.join(
      f'{k}="{v}"' for k, v in self.props.items())} style="{self.style}">
       {self.format_children()}
    </{self.name}>
    """

  def __call__(self, *children):
    self.children = list(children)
    return self

  def render(self):
    from IPython.core.display import HTML
    return HTML(self.html())

  def format_children(self):
    return ' '.join(str(c) for c in self.children)

  def display(self):
    from IPython.core.display import display
    self._display_handle = display(self.render(), display_id=True)

  def update(self):
    if self._display_handle:
      self._display_handle.update(self.render())


def prop(name):
  def g(self):
    return getattr(self, f'_{name}')

  def s(self, val):
    setattr(self, f'_{name}', val)
    self.update()

  return property(g, s)


class ReactiveNodeMeta(ABCMeta):
  def __new__(mcls, name, bases, namespace):
    annotations = namespace.get('__annotations__', {})

    props = set()
    prop_defaults = {}
    for name, annotation in annotations.items():
      if annotation is prop:
        if name in namespace:
          prop_defaults[name] = namespace[name]
        namespace[name] = prop(name)
        props.add(name)
    if '_props' in namespace:
      namespace['_props'].update(props)
    else:
      namespace['_props'] = props

    if '_default_props' in namespace:
      namespace['_default_props'].update(prop_defaults)
    else:
      namespace['_default_props'] = prop_defaults

    return super().__new__(mcls, name, bases, namespace)

class ReactiveMixin(metaclass=ReactiveNodeMeta):
  _props: set
  _default_props: dict

  def __init__(self, *args, **props):
    super(ReactiveMixin, self).__init__(*args, **props)
    self._init_props(**props)

  def _init_props(self, **passed_props):
    for prop in self._props:
      if prop not in passed_props and prop in self._default_props:
        setattr(self, prop, self._default_props[prop])
      elif prop in passed_props:
        setattr(self, prop, passed_props[prop])


class EmptyNode(Node):
  CHILDREN = True


class div(Node): pass


class span(Node): pass


class img(EmptyNode): pass
class p(Node): pass
class h1(Node): pass
class h2(Node): pass
class h3(Node): pass
class h4(Node): pass
class progress(EmptyNode): pass


class matplotfig(img):
  NAME = 'img'

  def __init__(self, figure, live=False, style=None, **props):
    super(matplotfig, self).__init__(style=style, **props)

    self.figure = figure
    self.live = live

    if not self.live:
      from .plot import figure_to_base64
      self.props['src'] = figure_to_base64(figure, data_encoded=True)


  def html(self):
    if self.live:
      from .plot import figure_to_base64
      self.props['src'] = figure_to_base64(self.figure, data_encoded=True)
    if 'src' not in self.props:
      from .plot import figure_to_base64
      self.props['src'] = figure_to_base64(self.figure, data_encoded=True)

    return super(matplotfig, self).html()


def _cell(val, size=25):
  return div(
    style=f'width: {size}px; height: {size}px; display: inline-block; background-color: rgba({50 + val}, {20 + val * .8}, {80 + val * .6}, 1); margin: 0; border: 1px solid rgba(0,0,0, .05)')


def _row(*args):
  return div(*args, style='margin: 0; padding: 0; font-size:0; white-space: nowrap;')


def tensor(t, cell_size=15, scaled=None, min=None, max=None):
  if t.numel() > 50000:
    raise ValueError('tensor too large')
  if scaled is None:
    max = t.max() if max is None else max
    min = t.min() if min is None else min
    scaled = (t.float() - min) / (max-min) * 255

  if t.ndim == 1:
    return div(
      f"shape: {tuple(t.shape)}",
      _row(*(_cell(v, size=cell_size) for v in scaled))
    )

  if t.ndim == 2:
    return div(
      f"shape: {tuple(t.shape)}",
      *[
        _row(*(_cell(v, size=cell_size) for v in row)) for row in scaled
      ]
    )

  if t.ndim >= 3:
    return div(
      f"shape: {tuple(t.shape)}",
      div(
        *(div(style='border: 1px solid #CCC; padding: 5px; margin: 3px; display: inline-block; border-radius: 6px')(
          f"index: {n}", tensor(x, scaled=x, cell_size=cell_size, min=min, max=max)
        ) for n, x in enumerate(scaled)
        )
      )
    )