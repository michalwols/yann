import torch
import torch.nn as nn


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

  def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob
    self.scale_by_keep = scale_by_keep

  def forward(self, x):
    if self.drop_prob == 0.0 or not self.training:
      return x
    keep_prob = 1 - self.drop_prob
    shape = (x.shape[0],) + (1,) * (
      x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and self.scale_by_keep:
      random_tensor.div_(keep_prob)
    return x * random_tensor

  def extra_repr(self):
    return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MlpHead(nn.Module):
  def __init__(
    self,
    dim,
    num_classes=1000,
    mlp_ratio=4,
    act_type='relu',
    drop_rate=0.2,
  ):
    super().__init__()
    hidden_features = min(2048, int(mlp_ratio * dim))
    self.fc1 = nn.Linear(dim, hidden_features, bias=False)
    self.norm = nn.BatchNorm1d(hidden_features)
    self.act = Act(hidden_features, act_type)
    self.drop = nn.Dropout(drop_rate)
    self.fc2 = nn.Linear(hidden_features, num_classes, bias=False)

  def forward(self, x):
    x = self.fc1(x)
    x = self.norm(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    return x


class Swish(nn.Module):
  def forward(self, x):
    return x * torch.sigmoid(x)


class HardSwish(nn.Module):
  def forward(self, x):
    return x * nn.Hardsigmoid(inplace=True)(x)


class GSiLU(nn.Module):
  def forward(self, x):
    return x * torch.sigmoid(x.mean((2, 3), keepdim=True))


class Act(nn.Module):
  def __init__(self, out_planes=None, act_type='relu', inplace=True):
    super(Act, self).__init__()

    self.act = None
    if act_type == 'relu':
      self.act = nn.ReLU(inplace=inplace)
    elif act_type == 'prelu':
      self.act = nn.PReLU(out_planes)
    elif act_type == 'swish':
      self.act = Swish()
    elif act_type == 'hardswish':
      self.act = HardSwish()
    elif act_type == 'hardsilu':
      self.act = nn.Hardswish(inplace=True)
    elif act_type == 'silu':
      self.act = nn.SiLU(inplace=True)
    elif act_type == 'gsilu':
      self.act = GSiLU()
      # self.act = SE(out_planes)
    elif act_type == 'gelu':
      self.act = nn.GELU()

  def forward(self, x):
    if self.act is not None:
      x = self.act(x)
    return x


class SE(nn.Module):
  def __init__(self, dim, ratio=4):
    super().__init__()
    hidden_dim = max(8, dim // ratio)

    self.se = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
      nn.BatchNorm2d(hidden_dim),
      nn.ReLU(),
      nn.Conv2d(hidden_dim, dim, kernel_size=1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return x * self.se(x)


class ConvX(nn.Module):
  def __init__(
    self,
    in_planes,
    out_planes,
    groups=1,
    kernel_size=3,
    stride=1,
    act_type='relu',
  ):
    super(ConvX, self).__init__()
    self.conv = nn.Conv2d(
      in_planes,
      out_planes,
      kernel_size=kernel_size,
      stride=stride,
      groups=groups,
      padding=kernel_size // 2,
      bias=False,
    )
    self.norm = nn.BatchNorm2d(out_planes)
    self.act = Act(out_planes, act_type)

  def forward(self, x):
    out = self.norm(self.conv(x))
    out = self.act(out)
    return out


class DropBottleNeck(nn.Module):
  def __init__(
    self,
    in_planes,
    out_planes,
    group_width=1,
    stride=1,
    act_type='relu',
    s_act_type=None,
    mlp_ratio=1.0,
    use_le=False,
    drop_path=0.0,
  ):
    super(DropBottleNeck, self).__init__()
    mid_planes = int(out_planes * mlp_ratio)

    self.stride = stride
    if stride == 2:
      self.ln = nn.LayerNorm(in_planes)
    self.le = (
      ConvX(
        in_planes,
        in_planes,
        groups=in_planes // group_width,
        kernel_size=3,
        act_type=None,
      )
      if use_le
      else nn.Identity()
    )
    self.conv_in = ConvX(
      in_planes,
      mid_planes,
      groups=1,
      kernel_size=1,
      stride=1,
      act_type=act_type,
    )
    self.conv = ConvX(
      mid_planes,
      mid_planes,
      groups=mid_planes // group_width,
      kernel_size=3,
      stride=stride,
      act_type=s_act_type,
    )
    # self.conv = nn.AvgPool2d(3, stride, 1)
    self.conv_out = ConvX(
      mid_planes,
      out_planes,
      groups=1,
      kernel_size=1,
      stride=1,
      act_type=None,
    )

    self.skip = nn.Identity()
    if stride == 2 and in_planes != out_planes:
      self.skip = nn.Sequential(
        ConvX(
          in_planes,
          in_planes,
          groups=in_planes // group_width,
          kernel_size=3,
          stride=stride,
          act_type=None,
        ),
        ConvX(
          in_planes,
          out_planes,
          groups=1,
          kernel_size=1,
          stride=1,
          act_type=None,
        ),
      )
    self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

  def forward(self, x):
    if self.stride == 2:
      x = self.ln(x.transpose(1, 3)).transpose(1, 3)
    out = self.le(x)
    out = self.conv_in(out)
    out = self.conv(out)
    out = self.conv_out(out)

    return self.drop_path(out) + self.skip(x)


class SFCNN(nn.Module):
  # pylint: disable=unused-variable
  def __init__(
    self,
    dims,
    layers,
    group_widths=1,
    block=DropBottleNeck,
    act_type='relu',
    s_act_type=None,
    mlp_ratio=1.0,
    use_le=False,
    drop_path_rate=0.0,
    num_classes=1000,
  ):
    super(SFCNN, self).__init__()
    self.block = block
    self.act_type = act_type
    self.s_act_type = act_type if s_act_type is None else s_act_type
    self.mlp_ratio = mlp_ratio
    self.use_le = use_le
    self.drop_path_rate = drop_path_rate

    if isinstance(dims, int):
      dims = [dims // 2, dims, dims * 2, dims * 4, dims * 8]
    else:
      dims = [dims[0] // 2] + dims

    if isinstance(group_widths, int):
      group_widths = [group_widths, group_widths, group_widths, group_widths]

    self.first_conv = ConvX(3, dims[0], 1, 3, 2, act_type=act_type)

    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]

    self.layer1 = self._make_layers(
      dims[0],
      dims[1],
      group_widths[0],
      layers[0],
      stride=2,
      drop_path=dpr[: layers[0]],
    )
    self.layer2 = self._make_layers(
      dims[1],
      dims[2],
      group_widths[1],
      layers[1],
      stride=2,
      drop_path=dpr[layers[0] : sum(layers[:2])],
    )
    self.layer3 = self._make_layers(
      dims[2],
      dims[3],
      group_widths[2],
      layers[2],
      stride=2,
      drop_path=dpr[sum(layers[:2]) : sum(layers[:3])],
    )
    self.layer4 = self._make_layers(
      dims[3],
      dims[4],
      group_widths[3],
      layers[3],
      stride=2,
      drop_path=dpr[sum(layers[:3]) : sum(layers[:4])],
    )

    # self.gap = nn.AdaptiveAvgPool2d(1)
    # self.classifier = MlpHead(dims[4], num_classes, act_type=act_type)

    head_dim = max(1024, dims[4])
    self.head = ConvX(dims[4], head_dim, 1, 1, 1, act_type=act_type)
    self.gap = nn.AdaptiveAvgPool2d(1)
    self.classifier = MlpHead(head_dim, num_classes, act_type=act_type)

    self.init_params(self)

  def _make_layers(
    self,
    inputs,
    outputs,
    group_width,
    num_block,
    stride,
    drop_path,
  ):
    layers = [
      self.block(
        inputs,
        outputs,
        group_width,
        stride,
        self.act_type,
        self.s_act_type,
        self.mlp_ratio,
        self.use_le,
        drop_path[0],
      ),
    ]

    for i in range(1, num_block):
      layers.append(
        self.block(
          outputs,
          outputs,
          group_width,
          1,
          self.act_type,
          self.s_act_type,
          self.mlp_ratio,
          self.use_le,
          drop_path[i],
        ),
      )

    return nn.Sequential(*layers)

  def init_params(self, model):
    for name, m in model.named_modules():
      if isinstance(m, nn.Conv2d):
        if 'first' in name:
          nn.init.normal_(m.weight, 0, 0.01)
        else:
          nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0001)
        nn.init.constant_(m.running_mean, 0)
      elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0.0001)
        nn.init.constant_(m.running_mean, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.first_conv(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    out = self.head(x)
    out = self.gap(out).flatten(1)
    out = self.classifier(out)
    return out
