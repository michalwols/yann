import dataclasses
from typing import Any, Union

import torch


def weighted_sum(tensors, weights):
  if len(tensors) < 2:
    raise ValueError('must pass at least 2 tensors')
  s = tensors[0] * weights[0]
  for t, w in zip(tensors[1:], weights[1:]):
    s.add_(w, t)
  return s


def one_hot(
  targets: torch.Tensor,
  num_classes=None,
  device=None,
  dtype=None,
  normalize=False,
):
  if torch.is_tensor(targets):
    if len(targets.shape) == 1:
      num = targets.shape[0]
      hot = torch.zeros(
        num,
        num_classes,
        device=device or targets.device,
        dtype=dtype,
      )
      hot.scatter_(1, targets.unsqueeze(1), 1.0)
      return hot
    elif len(targets.shape) == 2:
      pass

    raise ValueError('only dim 1 tensors supported')


def show_hist(hist):
  """Generates a simple block character histogram string."""
  chars = '  ▂▃▄▅▆▇█'
  # Handle empty histogram case
  if not hist:
    return '(empty)'
  top = max(hist) if hist else 0
  # Prevent division by zero if top is 0
  step = (top / float(len(chars) - 1)) if top > 0 else 1
  # Ensure step is at least 1 to avoid division by zero if counts are very small
  step = max(step, 1)
  return ''.join(chars[min(int(round(count / step)), len(chars) - 1)] for count in hist)


def tensor_histogram_stats(
  tensor: torch.Tensor,
  bins: int = 10,
  indent: str = '',
  min_val_item: float = None,
  max_val_item: float = None,
) -> str:
  """Generates a formatted string representation of a tensor's histogram.

  Includes a block character overview and detailed bins with counts, percentages,
  and inline bars. Handles integer tensors with smarter binning.

  Args:
      tensor: The tensor to analyze.
      bins: Default number of bins for histogram.
      indent: Indentation string for formatting.
      min_val_item: Pre-calculated minimum value (as Python number). Optional.
      max_val_item: Pre-calculated maximum value (as Python number). Optional.

  Returns:
      A formatted string representing the histogram.
  """
  # Recalculate min/max items if not provided (needed for standalone use or if failed before)
  if min_val_item is None or max_val_item is None:
    try:
      min_val_item = tensor.min().item()
      max_val_item = tensor.max().item()
    except RuntimeError:
      min_val_item = None
      max_val_item = None

  try:
    # Check applicability and edge cases first
    is_float = tensor.is_floating_point()
    is_complex = torch.is_complex(tensor)
    is_signed_int = tensor.dtype in (
      torch.int8,
      torch.int16,
      torch.int32,
      torch.int64,
    )
    is_unsigned_int = tensor.dtype == torch.uint8

    if not (is_float or is_complex or is_signed_int or is_unsigned_int):
      return f'{indent}  hist: (not applicable for this dtype)'

    if tensor.numel() == 0:
      return f'{indent}  hist: (empty tensor)'
    if torch.isnan(tensor.float()).any():  # Use float for isnan check
      return f'{indent}  hist: (contains NaN)'
    if (
      min_val_item is not None
      and max_val_item is not None
      and min_val_item == max_val_item
    ):
      return f'{indent}  hist: (all values ≈ {min_val_item:.4f})'
    if (
      min_val_item is None or max_val_item is None
    ):  # Check again after potential recalc
      return f'{indent}  hist: (could not determine min/max)'

    # Prepare tensor for histc
    float_tensor = tensor.float() if not (is_float or is_complex) else tensor

    # --- Smart Histogram Logic ---
    num_bins = bins
    histc_min = min_val_item
    histc_max = max_val_item
    label_format = 'range'

    is_integer_tensor = is_signed_int or is_unsigned_int
    if is_integer_tensor:
      num_unique_values = torch.unique(tensor).numel()
      value_range = int(max_val_item - min_val_item + 1)
      if num_unique_values <= 25 and num_unique_values == value_range:
        num_bins = num_unique_values
        histc_min = min_val_item - 0.5
        histc_max = max_val_item + 0.5
        label_format = 'int'

    h_counts = (
      float_tensor.histc(bins=num_bins, min=histc_min, max=histc_max).int().tolist()
    )

    max_h = max(h_counts) if h_counts else 0
    bar_max_width = 20
    bar_char = '█'

    block_hist = show_hist(h_counts)
    bin_width = (histc_max - histc_min) / num_bins
    total_count = float_tensor.numel()
    hist_lines = [f'hist: {block_hist}']
    for i, count in enumerate(h_counts):
      percentage = (count / total_count) * 100 if total_count > 0 else 0
      if label_format == 'int':
        label = f'{int(min_val_item + i):<5d}'
        label_width = 8
      else:
        bin_start = histc_min + i * bin_width
        bin_end = bin_start + bin_width
        prec = 3 if max(abs(bin_start), abs(bin_end)) < 10 else 2
        label = f'{bin_start:>{prec + 4}.{prec}f} - {bin_end:>{prec + 4}.{prec}f}'
        label_width = 2 * (prec + 4) + 3

      line_text = f'{label:<{label_width}} : {count:<8} ({percentage:5.1f}%)'
      bar_len = int(round((count / max_h) * bar_max_width)) if max_h > 0 else 0
      bar_str = bar_char * bar_len
      hist_lines.append(f'{line_text} | {bar_str}')
    return '\n'.join(f'{indent}  {line}' for line in hist_lines)

  except RuntimeError as e:
    return f'{indent}  hist: (error: {e})'


def describe_tensor(tensor: torch.Tensor, bins=10, indent: str = '') -> str:
  """Describes a single tensor."""
  try:
    stats = f'mean: {tensor.mean():.4f} std: {tensor.std():.4f} '
  except (
    RuntimeError
  ):  # Handle cases like boolean tensors where mean/std are not defined
    stats = ''

  try:
    min_val = tensor.min()
    max_val = tensor.max()
    sum_val = tensor.sum()
    # Extract item() immediately after successful calculation
    min_val_item = min_val.item()
    max_val_item = max_val.item()
    min_max_sum = (
      f'min: {min_val_item:.4f}  max: {max_val_item:.4f}  sum: {sum_val:.4f}'
    )
  except RuntimeError:  # Handle non-numeric types
    min_val_item = None  # Set items to None if calculation failed
    max_val_item = None
    min_max_sum = ''

  # Calculate histogram stats using the new helper function
  hist_str = tensor_histogram_stats(
    tensor,
    bins,
    indent,
    min_val_item,
    max_val_item,
  )

  # Limit tensor display for large tensors
  tensor_str = str(tensor)
  if len(tensor_str) > 1000:
    tensor_str = tensor_str[:500] + '\n... (tensor truncated) ...\n' + tensor_str[-500:]

  return f"""{indent}shape: {tuple(tensor.shape)} dtype: {tensor.dtype} device: {tensor.device} grad: {tensor.requires_grad} size: {tensor.numel() * tensor.element_size() / (1e6):,.5f} MB
{indent}{min_max_sum}  {stats}
{indent}{hist_str}

{indent}{tensor_str}
"""


def describe(data: Any, bins=10, indent: str = '') -> str:
  """
  Recursively describes tensors found in nested structures like lists, tuples, dicts, and dataclasses.
  """
  if torch.is_tensor(data):
    return describe_tensor(data, bins, indent)
  elif isinstance(data, dict):
    items_desc = []
    for k, v in data.items():
      items_desc.append(
        f"{indent}  '{k}':\n{describe(v, bins, indent + '    ')}",
      )
    return f'{indent}{{\n' + '\n'.join(items_desc) + f'\n{indent}}}'
  elif isinstance(data, (list, tuple)):
    is_list = isinstance(data, list)
    items_desc = [describe(item, bins, indent + '  ') for item in data]
    start, end = ('[', ']') if is_list else ('(', ')')
    return f'{indent}{start}\n' + '\n'.join(items_desc) + f'\n{indent}{end}'
  elif dataclasses.is_dataclass(data) and not isinstance(data, type):
    items_desc = []
    for field in dataclasses.fields(data):
      value = getattr(data, field.name)
      items_desc.append(
        f'{indent}  {field.name}:\n{describe(value, bins, indent + "    ")}',
      )
    return (
      f'{indent}{data.__class__.__name__}(\n' + '\n'.join(items_desc) + f'\n{indent})'
    )
  else:
    # For other types, just return their string representation
    return f'{indent}{str(data)}'
