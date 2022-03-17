import pytest
import torch

@pytest.fixture
def get_device():
    """Pytest fixture which simply returns the optimal device for testing models
    Args:
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device
