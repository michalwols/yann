from torch import nn
from torch.nn import init


def kaiming(model: nn.Module):
  for module in model.modules():
    if isinstance(module, nn.Conv2d):
      init.kaiming_normal(module.weight, mode='fan_out')
      if module.bias is not None:
        init.constant(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
      init.constant(module.weight, 1)
      init.constant(module.bias, 0)

    elif isinstance(module, nn.Linear):
      init.normal(module.weight, std=1e-3)
      if module.bias is not None:
        init.constant(module.bias, 0)


msr = kaiming
