from scipy.misc import imresize



def draw_mask(img, mask, blend=.5, cmap=None, interp='cubic'):
    if not cmap:
      import matplotlib.pylab as plt
      cmap = plt.get_cmap('jet')
    if isinstance(cmap, str):
      import matplotlib.pylab as plt
      cmap = plt.get_cmap(cmap)

    if mask.shape[:2] != img.shape[:2]:
        mask = imresize(mask, img.shape[:2], interp=interp)
    return (cmap(mask)[:,:,:3] * 255 * blend + img * (1-blend)).round().astype('uint8')


def class_activation_maps(features, weights, classes=None, normalize=True):
  num_samples, num_channels, rows, cols = features.shape

  classes = classes or list(range(weights.shape[0]))

  maps = []
  for sample in features:
    class_maps = {}
    for c in classes:
      blended_channels = (
            weights[c] @ sample.reshape(num_channels, rows * cols)).reshape(
        rows, cols)
      if normalize:
        x = blended_channels - blended_channels.min()
        x = x / x.max() * 255
        class_maps[c] = x.cpu().detach().numpy()
      else:
        class_maps[c] = blended_channels.cpu().detach().numpy()
    maps.append(class_maps)
  return maps


#
# class ClassActivationMaps:
#   def __init__(self):
#     self.conv_hook_handle = None
#     self.conv_features = None
#     self.cams = None
#
#   def hook(self, last_conv):
#     self.conv_hook_handle = last_conv.register_forward_hook()
#
#   def record_features(self, module, inputs, outputs):
#     self.conv_features = outputs
#
#   def draw(self, img):
#     pass
#
#   def show(self):
#     pass