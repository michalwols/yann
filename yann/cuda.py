import torch



def device_info(d=None):
  d = d or torch.cuda.current_device()
  return {
    'name': torch.cuda.get_device_name(d),
    'capability': torch.cuda.get_device_capability(d),
    'allocated_memory': torch.cuda.memory_allocated(d),
    'cached_memory': torch.cuda.memory_cached(d),
  }


sync = torch.cuda.synchronize