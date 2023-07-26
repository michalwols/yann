from torchvision.datasets import CocoDetection
from yann.data import Classes
from yann import default

import os

class CocoMultilabel(CocoDetection):
  def __init__(
      self,
      root=None,
      annFile=None
  ):
    root = root or default.datasets_root / 'mscoco/train2017'
    annFile = annFile or default.datasets_root / 'mscoco/annotations/instances_train2017.json'

    super(CocoMultilabel, self).__init__(
      root=root,
      annFile=annFile,
    )
    self.cat2cat = dict()
    for cat in self.coco.cats.keys():
      self.cat2cat[cat] = len(self.cat2cat)
    # print(self.cat2cat)

    self.category_id_to_name = {
      x['id']: x['name']
      for x in self.coco.cats.values()
    }

    self.labels = {}  # map from image id to label name
    for image_id in self.ids:
      annotation_ids = self.coco.getAnnIds(imgIds=image_id)
      annotations = self.coco.loadAnns(annotation_ids)
      self.labels[image_id] = list({
        self.category_id_to_name[annotation['category_id']]
        for annotation in annotations
      })

    self.classes = Classes.from_labels(self.labels.values())

  def __getitem__(self, index):
    img_id = self.ids[index]
    file_name = self.coco.loadImgs(img_id)[0]['file_name']
    path = os.path.join(self.root, file_name)
    return path, self.labels[img_id]