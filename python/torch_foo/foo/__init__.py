from typing import Optional, Union, List
import torch
import torch_foo._C as _C

# The backend module can be logically divided into a few blocks.
# - the Minimal API necessary to define a backend module in pytorch
# - the random API necessary to support setting seeds
# - the AMP api necessary to support automatic mixed precision

# Minimal API
def is_available() -> bool:
    r"""Returns a bool indicating if Foo is currently available."""
    if not hasattr(_C, "device_count"):
        return False
    return device_count() > 0

def current_device() -> int:
    r"""Returns the index of the currently selected device"""
    return _C.current_device()

# Random API
_cached_device_count: Optional[int] = None
def device_count() -> int:
    r"""Returns the number of Foos available"""
    global _cached_device_count
    if _cached_device_count is not None:
        return _cached_device_count
    count = _C._get_device_count()
    # in the future, we we add lazy initialization make sure
    # to check that we are initialized before caching the device count
    _cached_device_count = count
    return count

def _is_in_bad_fork() -> bool:
    r"""True if now in bad_fork, else False"""
    raise NotImplementedError

def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers for the devices"""
    raise NotImplementedError

def get_rng_state(device: Union[int, str, torch.device] = 'foo') -> torch.Tensor:
    r"""Returns a list of ByteTensor representing the random
        number states of all devices"""
    raise NotImplementedError

def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = 'foo') -> None:
    r"""Sets the random number generator state of the specified device"""
    raise NotImplementedError

# AMP API
def get_amp_supported_dtype() -> List[torch.dtype]:
    r"""Get the supported dtypes on your device in AMP"""
    raise NotImplementedError
