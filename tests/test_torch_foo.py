import torch
from torch_foo import add, multiply
import pytest

def test_add():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = add(a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(result, expected)

    # Test with different shapes (broadcasting)
    c = torch.tensor([10.0])
    result = add(a, c)
    expected = torch.tensor([11.0, 12.0, 13.0])
    assert torch.allclose(result, expected)

def test_multiply():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = multiply(a, b)
    expected = torch.tensor([4.0, 10.0, 18.0])
    assert torch.allclose(result, expected)
