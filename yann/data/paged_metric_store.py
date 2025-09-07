import torch
from typing import Dict, Optional, Union
from .paged_tensor import PagedTensor

class PagedMetricStore:
    """
    Stores time-series metrics using PagedTensor.

    Allows storing metrics that are added incrementally during training.
    Optionally supports moving older pages of tensor data from GPU to CPU
    to conserve GPU memory.
    """
    def __init__(
        self,
        page_size: int = 1024,
        default_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda:0' if torch.cuda.is_available() else 'cpu',
        cpu_paging: bool = True
    ):
        """
        Initializes the PagedMetricStore.

        Args:
            page_size: The number of elements each internal tensor page holds.
            default_dtype: The default data type for metrics if not inferred.
            device: The primary device (e.g., 'cuda:0' or 'cpu') where the
                    *active* page of metrics will reside.
            cpu_paging: If True and device is CUDA, move completed pages
                        (except the most recent one) to CPU asynchronously.
        """
        self.page_size = page_size
        self.default_dtype = default_dtype
        self.device = torch.device(device)
        self.cpu_device = torch.device('cpu')
        self.cpu_paging = cpu_paging and self.device.type == 'cuda' # Only page if primary device is CUDA

        # Store metric data: Dict[metric_name, PagedTensor]
        self.values: Dict[str, PagedTensor] = {}
        # Store corresponding steps (typically integers) - kept on CPU
        self.steps = PagedTensor(
            page_size=page_size,
            dtype=torch.int64,
            device=self.cpu_device
        )

    def update(self, step: Optional[int] = None, **metrics: Dict[str, Union[torch.Tensor, float, int]]):
        """
        Adds a new set of metric values at a given step.

        Args:
            step: The step index for this update. If None, uses the next sequential index.
            **metrics: Keyword arguments where the key is the metric name and
                       the value is the metric value (scalar or Tensor).
        """
        current_step = step if step is not None else len(self.steps)
        self.steps.append(torch.tensor(current_step, dtype=torch.int64)) # Steps always on CPU

        for name, value in metrics.items():
            # Convert value to tensor if it's not already
            if not isinstance(value, torch.Tensor):
                # Try to infer dtype, fallback to default
                dtype_to_use = getattr(value, 'dtype', self.default_dtype)
                try:
                    # Ensure value is compatible with tensor creation
                    value_tensor = torch.tensor(value, dtype=dtype_to_use)
                except TypeError:
                    # Fallback for non-numeric types if necessary, or raise error
                    value_tensor = torch.tensor(value, dtype=self.default_dtype) # Re-attempt with default
            else:
                value_tensor = value

            # Ensure value is on the primary device before potentially appending
            value_tensor = value_tensor.to(self.device)

            if name not in self.values:
                # Initialize PagedTensor for the new metric
                self.values[name] = PagedTensor(
                    page_size=self.page_size,
                    dtype=value_tensor.dtype,
                    device=self.device, # New pages start on primary device
                    element_shape=value_tensor.shape
                )

            paged_tensor = self.values[name]

            # --- CPU Paging Logic ---
            # Check if we are about to complete the last page AND cpu_paging is enabled
            if self.cpu_paging and paged_tensor.pages and paged_tensor.current_idx_in_page == paged_tensor.page_size -1:
                # The page that is *about* to become full is the last one
                page_to_move = paged_tensor.pages[-1]
                if page_to_move.device != self.cpu_device:
                    # Move the tensor page to CPU asynchronously.
                    # Directly modify the list within PagedTensor.
                    # Note: This assumes the page isn't needed immediately on CPU.
                    paged_tensor.pages[-1] = page_to_move.to(self.cpu_device, non_blocking=True)
                    # print(f"Step {current_step}: Moving page {len(paged_tensor.pages)-1} of metric '{name}' to CPU") # Debugging

            # Append the value (potentially creating a new page on self.device)
            paged_tensor.append(value_tensor)

    def __getitem__(self, name: str) -> PagedTensor:
        """Gets the PagedTensor object for a given metric name."""
        if name not in self.values:
            raise KeyError(f"Metric '{name}' not found in store.")
        return self.values[name]

    def __contains__(self, name: str) -> bool:
        """Checks if a metric name exists in the store."""
        return name in self.values

    def __len__(self) -> int:
        """Returns the number of steps recorded."""
        return len(self.steps)

    def keys(self):
        """Returns the names of the metrics stored."""
        return self.values.keys()

    def get_steps_tensor(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns all recorded steps as a single tensor on the target device (or CPU)."""
        return self.steps.to_tensor(target_device=target_device or self.cpu_device)

    def get_metric_tensor(self, name: str, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Gets the full data for a metric as a single tensor on the target device.

        Args:
            name: The name of the metric.
            target_device: The device for the resulting tensor (defaults to CPU).

        Returns:
            A single tensor containing all data points for the metric.

        Raises:
            KeyError: If the metric name is not found.
        """
        if name not in self.values:
            raise KeyError(f"Metric '{name}' not found in store.")
        # PagedTensor.to_tensor defaults to CPU if target_device is None
        return self.values[name].to_tensor(target_device=target_device or self.cpu_device)

    def to_dict(self, target_device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Converts the entire store to a dictionary of flat tensors.

        Args:
            target_device: The device for the resulting tensors (defaults to CPU).

        Returns:
            A dictionary mapping metric names to their full data tensors.
        """
        target_dev = target_device or self.cpu_device
        return {name: pt.to_tensor(target_device=target_dev) for name, pt in self.values.items()}

    def __repr__(self):
        metric_names = list(self.values.keys())
        return (f"PagedMetricStore(steps={len(self)}, metrics={metric_names}, "
                f"page_size={self.page_size}, device={self.device}, cpu_paging={self.cpu_paging})")

# Example Usage:
# if __name__ == '__main__':
#     store = PagedMetricStore(page_size=5, device='cuda:0', cpu_paging=True)
#     print(store)
#
#     # Simulate updates
#     for i in range(18):
#         store.update(
#             step=i,
#             loss=torch.rand(1).item() * (1 / (i + 1)), # Scalar
#             accuracy=torch.rand(1) + 0.5, # Tensor on CPU initially
#             some_vec = torch.randn(2, device='cuda:0') # Tensor on GPU
#         )
#         if i % 4 == 0:
#             print(f"Step {i}: Pages in 'loss': {[p.device for p in store['loss'].pages]}")
#             print(f"Step {i}: Pages in 'accuracy': {[p.device for p in store['accuracy'].pages]}")
#             print(f"Step {i}: Pages in 'some_vec': {[p.device for p in store['some_vec'].pages]}")
#
#
#     print(f"Total steps: {len(store)}")
#     print(f"Loss tensor (on CPU): {store.get_metric_tensor('loss')}")
#     print(f"Accuracy tensor (on GPU): {store.get_metric_tensor('accuracy', target_device=torch.device('cuda:0'))}")
#     print(f"Steps tensor (on CPU): {store.get_steps_tensor()}")
#
#     # Access using PagedTensor directly
#     loss_paged = store['loss']
#     print(f"Loss at step 5: {loss_paged[5]}") # Access single element
#     print(f"Loss slice [8:12]: {loss_paged[8:12]}") # Access slice
#     print(f"Device of loss slice [8:12]: {loss_paged[8:12].device}")
#
#     some_vec_paged = store['some_vec']
#     print(f"Vec slice [3:7]: {some_vec_paged[3:7]}")
#     print(f"Device of vec slice [3:7]: {some_vec_paged[3:7].device}") # Should be cuda:0
#
#     print(store)
#     # Convert all to CPU tensors
#     all_data_cpu = store.to_dict()
#     print({k: (v.shape, v.device) for k, v in all_data_cpu.items()}) 