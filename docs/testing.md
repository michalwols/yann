# Testing and Validation

```python
import torch
from yann.testing import check_tensor


t = torch.randn(32, 3, 224, 224)
t2 = t

check_tensor(
  t,
  device='cpu',
  gte=0,  # greater than or equal
  lte=1,  # less than or equal
  share_memory=t2,  # using the same storage
  shape=(None, 3, 224, 224),  # dimensions match
  anomalies=True,  # check for infs or NaNs
)
```