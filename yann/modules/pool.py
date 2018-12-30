from torch.nn import functional as F



def mac(batch):
  """MAC Pooling"""
  return F.adaptive_max_pool2d(batch, (1,1))

def spoc(batch):
  """SPoC Pooling"""
  return F.adaptive_avg_pool2d(batch, (1,1))


def generalized_mean(batch, p=3, eps=1e-8):
  """Generalized Mean Pooling (GeM)
  p=1 === spoc
  larger p ==> mac

  larger p leads to more localized (max) features
  """
  return F.adaptive_avg_pool2d(
    batch.clamp(min=eps) ** p,
    (1, 1)
  ) ** (1 / p)

gem = generalized_mean