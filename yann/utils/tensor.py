import torch

def weighted_sum(tensors, weights):
  if len(tensors) < 2:
    raise ValueError('must pass at least 2 tensors')
  s = tensors[0] * weights[0]
  for t, w in zip(tensors[1:], weights[1:]):
    s.add_(w, t)
  return s


def one_hot(targets: torch.Tensor, num_classes=None, device=None, dtype=None, normalize=False):
  if torch.is_tensor(targets):
    if len(targets.shape) == 1:
      num = targets.shape[0]
      hot = torch.zeros(num, num_classes, device=device or targets.device, dtype=dtype)
      hot.scatter_(1, targets.unsqueeze(1), 1.0)
      return hot
    elif len(targets.shape) == 2:
      pass

    raise ValueError('only dim 1 tensors supported')



def show_hist(hist):
  chars = '_▁▂▃▄▅▆▇█'
  top = max(hist)
  step = (top / float(len(chars) - 1)) or 1
  return ''.join(chars[int(round(count / step))] for count in hist)


def describe(tensor: torch.Tensor, bins=10) -> str:
  try:
    stats = (
      f"μ={tensor.mean():.4f}  σ={tensor.std():.4f}\n"
    )
  except:
    stats = ''

  try:
    h = tensor.histc(bins=bins).int().tolist()
    hist = (
      f"hist: {h}\n"
      f"hist: {show_hist(h)}\n"
    )
  except:
    hist = ''
  return (

    f"{tuple(tensor.shape)} "
    f"{tensor.dtype} "
    f"{tensor.device}"
    f"{' grad' if tensor.requires_grad else ''} "
    f"({tensor.numel() * tensor.element_size() / (1e6):,.5f} MB)\n"
    f"{stats}"
    f"{hist}"
    f"min: {tensor.min():.4f}  max: {tensor.max():.4f}  sum: {tensor.sum():.4f}\n\n"
  )