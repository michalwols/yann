from .html import div, prop, ReactiveMixin, Node


class ProgressBar(ReactiveMixin, Node):
  value: prop = 0
  max: prop = 100
  color: prop = 'lightgreen'
  background: prop = 'lightgray'

  def html(self):
    return (
      div(
        style=f"""
        position: relative;
        min-height: 20px; 
        border-radius: 6px; 
        background-color: {self.background}; 
        overflow: hidden; 
        padding: 5px 10px;
        """
      )(
        div(style='position:relative; z-index: 1;')(
          *self.children
        ),
        div(
          style=f"""
          background-color: {self.color};
          position: absolute;
          top: 0;
          bottom: 0;
          left: 0;
          right: {100 - self.value / self.max * 100}%;
          z-index: 0;
        """)
        )
    ).html()