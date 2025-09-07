import torch
import math

class PagedTensor:
    """
    Stores a sequence of tensors ("pages") to represent a larger logical tensor.
    Supports appending elements and flat indexing (__getitem__, __setitem__)
    as if it were a single contiguous tensor. Useful for collecting data
    incrementally without expensive reallocations.
    """
    def __init__(self, page_size: int, dtype: torch.dtype, device: torch.device, element_shape: tuple = ()):
        if not isinstance(page_size, int) or page_size <= 0:
            raise ValueError("page_size must be a positive integer")

        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self.element_shape = tuple(element_shape) # Shape of a single element
        self.page_shape = (self.page_size, *self.element_shape)

        self.pages = []           # List to hold the tensor pages
        self.current_idx_in_page = 0 # Index within the *last* page
        self.total_elements = 0   # Total number of elements stored across all pages

    def append(self, element: torch.Tensor):
        """Appends a single element tensor to the PagedTensor."""
        # Ensure element is a tensor and has the correct shape and type
        if not isinstance(element, torch.Tensor):
             element = torch.tensor(element, dtype=self.dtype, device=self.device)
        if element.shape != self.element_shape:
            raise ValueError(f"Element shape {element.shape} does not match expected shape {self.element_shape}")
        if element.dtype != self.dtype:
             # Try casting, warn or raise error if desired
             element = element.to(dtype=self.dtype)
        if element.device != self.device:
             element = element.to(device=self.device)


        # Add a new page if needed
        if not self.pages or self.current_idx_in_page == self.page_size:
            new_page = torch.empty(self.page_shape, dtype=self.dtype, device=self.device)
            self.pages.append(new_page)
            self.current_idx_in_page = 0

        # Add element to the last page
        self.pages[-1][self.current_idx_in_page] = element
        self.current_idx_in_page += 1
        self.total_elements += 1

    def __len__(self):
        """Returns the total number of elements stored."""
        return self.total_elements

    def _get_location(self, index: int) -> tuple[int, int]:
        """Calculates the page index and index within the page for a flat index."""
        if not isinstance(index, int):
             raise TypeError("Index must be an integer")
        if index < 0:
            index += self.total_elements
        if not (0 <= index < self.total_elements):
            raise IndexError(f"Index {index} out of range for size {self.total_elements}")

        page_idx = index // self.page_size
        idx_in_page = index % self.page_size
        return page_idx, idx_in_page

    def __getitem__(self, index):
        """Gets element(s) using flat indexing (integer or slice)."""
        if isinstance(index, int):
            page_idx, idx_in_page = self._get_location(index)
            # Return element directly from its page (could be CPU or GPU)
            return self.pages[page_idx][idx_in_page]
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.total_elements)
            indices = range(start, stop, step)
            if not indices:
                 return torch.empty((0, *self.element_shape), dtype=self.dtype, device=self.device) # Use primary device for empty

            # Determine target device (use device of the first element's page)
            first_page_idx, first_idx_in_page = self._get_location(indices[0])
            target_device = self.pages[first_page_idx].device

            results = []
            for i in indices:
                page_idx, idx_in_page = self._get_location(i)
                element = self.pages[page_idx][idx_in_page]
                # Move to target device if necessary
                if element.device != target_device:
                    element = element.to(target_device)
                results.append(element)

            # Stack results into a single tensor on the target_device
            if self.element_shape:
                 return torch.stack(results, dim=0)
            else:
                 return torch.tensor(results, dtype=self.dtype, device=target_device)
        else:
            raise TypeError(f"Indices must be integers or slices, not {type(index)}")

    def __setitem__(self, index, value):
        """Sets element(s) using flat indexing (integer or slice)."""
        if isinstance(index, int):
            page_idx, idx_in_page = self._get_location(index)
            # Ensure value is compatible
            if not isinstance(value, torch.Tensor):
                 value = torch.tensor(value, dtype=self.dtype, device=self.device)
            if value.shape != self.element_shape:
                 raise ValueError(f"Value shape {value.shape} incompatible with element shape {self.element_shape}")
            # Cast type/device if necessary (optional, could raise error)
            value = value.to(dtype=self.dtype, device=self.device)

            self.pages[page_idx][idx_in_page] = value

        elif isinstance(index, slice):
            start, stop, step = index.indices(self.total_elements)
            indices = range(start, stop, step)

            if not isinstance(value, torch.Tensor):
                 # Attempt to convert sequence to tensor
                 value = torch.tensor(value, dtype=self.dtype, device=self.device)

            if len(indices) != value.shape[0]:
                raise ValueError(f"Attempted to assign {value.shape[0]} elements to slice of size {len(indices)}")
            if value.shape[1:] != self.element_shape:
                 raise ValueError(f"Value tensor shape {value.shape[1:]} incompatible with element shape {self.element_shape}")

            # Cast type/device if necessary
            value = value.to(dtype=self.dtype, device=self.device)

            for i, val_idx in enumerate(indices):
                page_idx, idx_in_page = self._get_location(val_idx)
                self.pages[page_idx][idx_in_page] = value[i]
        else:
            raise TypeError(f"Indices must be integers or slices, not {type(index)}")

    def to_tensor(self, target_device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Concatenates all pages into a single flat tensor on the target_device.
        Warning: Can be memory-intensive if the total size is large.
        """
        if not self.pages:
            return torch.empty((0, *self.element_shape), dtype=self.dtype, device=target_device)

        # Handle potentially partially filled last page
        num_full_pages = len(self.pages) - 1
        tensors_to_cat = []

        # Process full pages
        for i in range(num_full_pages):
            page = self.pages[i]
            tensors_to_cat.append(page.to(target_device)) # Move page to target device

        # Process last page (potentially partial)
        if len(self.pages) > 0 and self.current_idx_in_page > 0:
             last_page_data = self.pages[-1][:self.current_idx_in_page]
             tensors_to_cat.append(last_page_data.to(target_device)) # Move partial page

        if not tensors_to_cat: # Should only happen if total_elements is 0
             return torch.empty((0, *self.element_shape), dtype=self.dtype, device=target_device)

        return torch.cat(tensors_to_cat, dim=0)


    def __repr__(self):
        return (f"PagedTensor(total_elements={self.total_elements}, page_size={self.page_size}, "
                f"num_pages={len(self.pages)}, element_shape={self.element_shape}, "
                f"dtype={self.dtype}, device={self.device})") 