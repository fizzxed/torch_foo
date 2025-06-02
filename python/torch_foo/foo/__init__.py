from typing import Optional
import torch_foo._C as _C

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


def is_available() -> bool:
    r"""Returns a bool indicating if Foo is currently available."""
    if not hasattr(_C, "device_count"):
        return False
    return device_count() > 0

def current_device() -> int:
    r"""Returns the index of the currently selected device"""
    return _C.current_device()
