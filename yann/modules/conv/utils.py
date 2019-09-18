import math


def get_tf_same_padding(
    tensor,
    kernel_size,
    stride=1,
    dilation=1
):
  if isinstance(kernel_size, int):
    return tuple(
      *tf_same_pad(tensor.shape[2], kernel_size, stride=stride, dilation=dilation),
      *tf_same_pad(tensor.shape[3], kernel_size, stride=stride, dilation=dilation)
    )
  else:
    return tuple(
      *tf_same_pad(tensor.shape[2], kernel_size[0], stride=stride, dilation=dilation),
      *tf_same_pad(tensor.shape[3], kernel_size[1], stride=stride, dilation=dilation)
    )

def tf_same_pad(size, kernel_size, stride=1, dilation=1):
  pad = max(
    0,
    (math.ceil(size / stride) - 1) * stride
      + (kernel_size - 1) * dilation
      + 1 - size
  )
  left_pad = pad // 2
  return left_pad, pad - left_pad


def get_same_padding(kernel_size, stride=1, dilation=1):
  return ((stride - 1) + dilation * (kernel_size - 1)) // 2