from io import BytesIO
from PIL import Image
from itertools import zip_longest
import base64


def base64str(img):
  s = BytesIO()
  img.save(s, format='JPEG')
  return (
      "data:image/jpeg;base64,"
      + base64.b64encode(s.getvalue()).decode("utf-8")
  )


def show_images(paths, labels=None, urls=None, w=400, h=400):
  from IPython.core.display import display, HTML
  from pathlib import Path

  if isinstance(paths[0], (tuple, list)):
    items = paths
  else:
    items = zip_longest(paths, labels or [])


  tags = []
  for x, l in items:
    if urls or (urls is not False and isinstance(x, (str, Path)) and str(x).startswith('http')):
      src = str(x)
    else:
      img = Image.open(x) if isinstance(x, str) else x
      img.thumbnail((w, h))
      src = base64str(img)

    if l:
      tags.append(
        f'''
        <div style="display: inline-block; padding: 3px">
          <img 
            style="max-width: {w}px; max-height: {h}px; margin: 3px;"
             src={src} />
          <p>{l if isinstance(l, str) else ', '.join(l)}</p>
        </div>
        '''
      )
    else:
      tags.append(
        f'''
        <div style="display: inline-block; padding: 3px">
          <img 
            style="max-width: {w}px; max-height: {h}px; margin: 3px;"
             src={src} />
        </div>
        '''
      )
  return display(HTML(f'<div>{"".join(tags)}</div>'))