import torch

def check_tensor(
    t: torch.Tensor,
    device=None,
    dtype=None,
    requires_grad=None,
    contiguous=None,
    pinned=None,
    shape=None,
    anomalies=True,
    equal=None,
    close=None,
    lt=None,
    gt=None,
    lte=None,
    gte=None,
    none=False
):
  if not none:
    assert t is not None
  if device:
    if isinstance(device, str):
      device = torch.device(device)
    assert t.device == device
  if dtype:
    assert t.dtype == dtype
  if requires_grad is not None:
    assert t.requires_grad == requires_grad
  if contiguous is not None:
    assert t.is_contiguous() == contiguous
  if pinned is not None:
    assert t.is_pinned() == pinned
  if shape:
    assert t.shape == torch.Size(shape)
  if anomalies:
    assert not torch.isnan(t).any() and not torch.isinf(t).any()
  if equal is not None:
    if not torch.is_tensor(equal):
      equal = torch.Tensor(equal, dtype=t.dtype, device=t.device)
    assert (t == equal).all()
  if close is not None:
    if not torch.is_tensor(close):
      close = torch.Tensor(close, dtype=t.dtype, device=t.device)
    assert torch.allclose(t, close)
  if lt is not None:
    assert (t < lt).all()
  if lte is not None:
      assert (t <= lte).all(), lte
  if gt is not None:
    assert (t > gt).all(), t > gte
  if gte is not None:
    assert (t >= gte).all(), t >= gte


# def profile(name=None, sync=True, max=None, track=False, log=True):
#   """
#   track execution speed of code block
#   Args:
#     sync: cuda synchronize or not
#     max: fail if takes longer than x amount of time
#     name:
#     track:
#
#   Returns:
#
#   """
#   pass
#
#
# def track_value():
#   pass

#
# with profile('forward pass', max=.3):
#   pass


# def assert_finite(x):
#   pass
#
#
# def check_stability():
#   pass
#
#
# def check_numerical_issues(function, shapes, types):
#   pass

