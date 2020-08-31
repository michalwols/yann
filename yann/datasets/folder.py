from torchvision.datasets.folder import ImageFolder as IF, default_loader

from yann.data import Classes


class ImageFolder(IF):
  def __init__(
    self,
    root,
    transform=None,
    target_transform=None,
    loader=default_loader,
    is_valid_file=None
  ):
    super(ImageFolder, self).__init__(
      root=root,
      transform=transform,
      target_transform=target_transform,
      loader=loader,
      is_valid_file=is_valid_file
    )

    self.classes = Classes(self.classes)