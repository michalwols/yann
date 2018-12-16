class styles(dict):
  def __init__(self, *args, **props):
    super().__init__()
    if args:
      if isinstance(args[0], dict):
        self.update(args[0])
    if isinstance(args[0], str):
      for l in args[0].split(';'):
        if l:
          k, v = l.split(':')

          self[k.replace('-', '_').strip()] = v.strip()

    self.update(props)

  def __str__(self):
    return ' '.join(f"{k.replace('_', '-')}: {v};"
                    for k, v in self.items())


class Node:
  NAME = None
  EMPTY = False

  def __init__(self, *children, style=None, **props):
    self.children = children
    self.props = props
    self.style = style if isinstance(style, styles) else styles(style)

  @property
  def name(self):
    return self.NAME or self.__class__.__name__

  def __str__(self):
    if self.EMPTY:
      return f"""
         <{self.name} {' '.join(
        f'{k}="{v}"' for k, v in self.props.items())}  style="{self.style}"/>
      """

    return f"""
    <{self.name} {' '.join(
      f'{k}="{v}"' for k, v in self.props.items())} style="{self.style}">
       {' '.join(str(c) for c in self.children)}
    </{self.name}>
    """

  def render(self, target='notebook'):
    if target == 'notebook':
      from IPython.core.display import display, HTML
      return display(HTML(str(self)))

    return str(self)


class div(Node): pass


class span(Node): pass


class img(Node):
  EMPTY = True


class p(Node): pass


class h1(Node): pass
