from collections import OrderedDict

from .tensor import weighted_sum


def average_state_dicts(dicts, weights=None):
  if not weights:
    weights = (1 / len(dicts),) * len(dicts)
  averaged = OrderedDict()
  for k in dicts[0]:
    averaged[k] = weighted_sum([d[k] for d in dicts], weights)
  return averaged
