import pytest
import torch

from triton_ops_lab.contracts import BoundTensorGuard
from triton_ops_lab.ops import prepare_matmul, UnsupportedBackend


def test_guard_allows_values_but_rejects_storage_swap():
    x = torch.randn(3, 7)
    guard = BoundTensorGuard((x,))
    x.copy_(torch.ones_like(x))
    guard.check()
    # .data replacement bypasses version counters; pointer checks must catch it.
    x.data = torch.zeros_like(x)
    with pytest.raises(RuntimeError, match="prepare"):
        guard.check()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.transpose_(0, 1),
        lambda x: x.resize_(21),
        lambda x: x.requires_grad_(True),
    ],
)
def test_guard_rejects_contract_changes(mutation):
    x = torch.randn(3, 7)
    guard = BoundTensorGuard((x,))
    mutation(x)
    with pytest.raises(RuntimeError, match="metadata/storage"):
        guard.check()


def test_preparation_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(UnsupportedBackend, match="no CPU fallback"):
        prepare_matmul(torch.ones(2, 3), torch.ones(3, 4))
