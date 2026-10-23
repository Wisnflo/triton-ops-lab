import platform

import pytest
import torch


def pytest_collection_modifyitems(items):
    available = (
        platform.system() == "Linux"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
    )
    for item in items:
        if "gpu" in item.keywords and not available:
            item.add_marker(
                pytest.mark.skip(
                    reason="Real Linux NVIDIA CUDA GPU (CC>=8.0) required; not verified on Mac"
                )
            )


@pytest.fixture(autouse=True)
def deterministic_inputs():
    torch.manual_seed(2026)
    torch.set_num_threads(2)
