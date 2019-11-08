import torch
from contextlib import contextmanager

from .utils.debug import iter_allocated_tensors
from .exceptions import CheckFailure


def check_tensor(
    t: torch.Tensor,
    device=None,
    dtype=None,
    requires_grad=None,
    contiguous=None,
    pinned=None,
    share_memory=None,
    not_share_memory=None,
    shape=None,
    same=None,
    different=None,
    anomalies=True,
    equal=None,
    close=None,
    like=None,
    lt=None,
    gt=None,
    lte=None,
    gte=None,
    none=False
):
  if share_memory is not None:
    assert t.storage().data_ptr() == share_memory.storage().data_ptr()
  if not_share_memory is not None:
    assert t.storage().data_ptr() != not_share_memory.storage().data_ptr()
  if different is not None:
    assert different is not t
  if same is not None:
    assert same is t
  if like is not None:
    check_tensor(t, device=like.device, shape=like.shape, dtype=like.shape)
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
    for x, y in zip(t.shape, shape):
      if y is not None:
        assert x == y
    assert len(t.shape) == len(shape)
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


@contextmanager
def newly_allocated_tensors(count=None, max=None):
  """
  Counts the number of allocated tensors at start and exit of block.
  Args:
    count: exact count
    max: at most this many
    new: if True will could newly allocated, otherwise will compare

  Returns:

  """

  initial = set(iter_allocated_tensors())
  new_tensors = set()
  yield new_tensors
  allocated = set(iter_allocated_tensors())
  new_tensors.update(allocated - initial)
  diff = len(new_tensors)

  if count is not None and count != diff:
    raise CheckFailure(f'Expected {count} tensor allocations but got {diff}')
  if max is not None:
    raise CheckFailure(f'Expected at most {count} tensor allocations but got {diff}')


class Checker:
  def newly_allocated_tensors(self, count=None, max=None):
    return newly_allocated_tensors(count=count, max=max)

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

