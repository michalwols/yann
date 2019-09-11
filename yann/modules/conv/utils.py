

def get_same_padding(kernel_size, stride=1, dilation=1):
  return ((stride - 1) + dilation * (kernel_size - 1)) // 2