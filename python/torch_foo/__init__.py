import os

# Disable autoloading before running `import torch` to avoid circular dependencies
ENV_AUTOLOAD = os.getenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch # Ensure torch is loaded first
# from torch_foo._C import add, multiply
from torch_foo._C import bar
from torch_foo import foo

__all__ = ['add', 'multiply']

# expose torch_foo.foo as torch.foo
torch._register_device_module("foo", foo)

# This function is an entrypoint called by PyTorch
# when running `import torch`. There is no need to do anything.
def _autoload():
    # We should restore the switch for sub processes
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = ENV_AUTOLOAD
    print("torch_foo autoloaded")
