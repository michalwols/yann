import collections
from typing import Any, Callable, Dict, List, Optional, Tuple

from torchvision.datasets.voc import ET_Element, ET_parse, _VOCBase

from yann.data import Classes


class VOCMultilabel(_VOCBase):
  _SPLITS_DIR = 'Main'
  _TARGET_DIR = 'Annotations'
  _TARGET_FILE_EXT = '.xml'

  def __init__(
    self,
    root: str,
    year: str = '2012',
    image_set: str = 'train',
    download: bool = False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
  ):
    super(VOCMultilabel, self).__init__(
      root=root,
      year=year,
      image_set=image_set,
      download=download,
      transform=transform,
      target_transform=target_transform,
      transforms=transforms,
    )

    self.classes = Classes.from_labels(x[1] for x in self)

  @property
  def annotations(self) -> List[str]:
    return self.targets

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is a dictionary of the XML tree.
    """
    img = self.images[index]
    target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
    target = list(set(a['name'] for a in target['annotation']['object']))

    return img, target

  def parse_voc_xml(self, node: ET_Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
      def_dic: Dict[str, Any] = collections.defaultdict(list)
      for dc in map(self.parse_voc_xml, children):
        for ind, v in dc.items():
          def_dic[ind].append(v)
      if node.tag == 'annotation':
        def_dic['object'] = [def_dic['object']]
      voc_dict = {
        node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()},
      }
    if node.text:
      text = node.text.strip()
      if not children:
        voc_dict[node.tag] = text
    return voc_dict
