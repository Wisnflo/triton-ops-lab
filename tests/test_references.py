import pytest
import torch
import torch.nn.functional as F

from triton_ops_lab import attention, conv2d, matmul, softmax, vector_add
from triton_ops_lab.models import TinyCNN, TinyTransformer
from triton_ops_lab.references import attention_explicit
from triton_ops_lab.validation import compare


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 23, 19), (65, 33, 49)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("epilogue", [False, True])
def test_gemm(dtype, shape, transpose, epilogue):
    m, n, k = shape
    a, b = (
        (torch.randn(k, m, dtype=dtype).T, torch.randn(n, k, dtype=dtype).T)
        if transpose
        else (torch.randn(m, k, dtype=dtype), torch.randn(k, n, dtype=dtype))
    )
    bias = torch.randn(n, dtype=dtype) if epilogue else None
    actual = matmul(a, b, bias, activation="gelu" if epilogue else None, backend="reference")
    expected = a.double() @ b.double()
    if epilogue:
        expected = F.gelu(expected + bias.double(), approximate="tanh")
    assert compare(actual, expected, "matmul")["passed"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_softmax_large_values(dtype):
    x = torch.randn(5, 37, dtype=dtype) * 100 + 10000
    actual = softmax(x, backend="reference")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.sum(-1), torch.ones(5, dtype=dtype))
    assert compare(actual, torch.softmax(x.double(), -1), "softmax")["passed"]
    equal = torch.full((2, 7), 10000.0, dtype=dtype)
    torch.testing.assert_close(softmax(equal, backend="reference"), torch.full_like(equal, 1 / 7))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n,d", [(1, 32), (35, 64), (67, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_online_attention(causal, n, d, dtype):
    q, k, v = [torch.randn(1, 2, n, d, dtype=dtype) for _ in range(3)]
    actual = attention(q, k, v, causal=causal, backend="reference")
    expected = attention_explicit(q.double(), k.double(), v.double(), causal)
    assert compare(actual, expected, "attention")["passed"]
    if causal:
        torch.testing.assert_close(actual[..., 0, :], v[..., 0, :])
        changed = v.clone()
        changed[..., -1, :] += 100
        altered = attention(q, k, changed, causal=True, backend="reference")
        torch.testing.assert_close(actual[..., :-1, :], altered[..., :-1, :])


def test_attention_extreme_logits():
    q, k, v = [torch.randn(1, 1, 37, 32, dtype=torch.float64) for _ in range(3)]
    q, k = q * 100, k * 100
    actual = attention(q, k, v, causal=True, backend="reference")
    torch.testing.assert_close(actual, attention_explicit(q, k, v, True), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("kernel", [1, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conv(kernel, stride, padding, dtype):
    x = torch.randn(2, 3, 7, 9, dtype=dtype)
    w = torch.randn(5, 3, kernel, kernel, dtype=dtype)
    actual = conv2d(x, w, stride=stride, padding=padding, backend="reference")
    expected = F.conv2d(x.double(), w.double(), stride=stride, padding=padding)
    assert actual.shape == (
        2,
        5,
        (7 + 2 * padding - kernel) // stride + 1,
        (9 + 2 * padding - kernel) // stride + 1,
    )
    assert compare(actual, expected, "conv2d")["passed"]


def test_conv_hand_computed():
    x = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    w = torch.ones(1, 1, 3, 3)
    assert conv2d(x, w, backend="reference").item() == 45


@pytest.mark.parametrize("kind", ["transformer", "cnn"])
def test_models(kind):
    model = TinyTransformer() if kind == "transformer" else TinyCNN()
    x = torch.randn(1, 7, 64) if kind == "transformer" else torch.randn(1, 3, 7, 9)
    expected = model(x, "torch")
    actual = model(x, "reference")
    assert compare(actual, expected, kind)["passed"]


def test_add_and_validation():
    a, b = torch.randn(259), torch.randn(259)
    torch.testing.assert_close(vector_add(a, b, backend="reference"), a + b)
    assert not compare(torch.tensor([float("nan")]), torch.ones(1))["passed"]
    assert not compare(torch.ones(2), torch.ones(3))["passed"]
    assert not compare(torch.ones(1), torch.zeros(1))["passed"]
    assert compare(torch.zeros(1), torch.zeros(1))["relative_l2"] == 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: vector_add(torch.ones(2), torch.ones(3)),
        lambda: vector_add(torch.ones(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64)),
        lambda: vector_add(torch.ones(2, requires_grad=True), torch.ones(2)),
        lambda: softmax(torch.ones(2, 3).T),
        lambda: matmul(torch.ones(2, 3), torch.ones(4, 5)),
        lambda: matmul(torch.ones(2, 3), torch.ones(3, 4), torch.ones(3)),
        lambda: matmul(torch.ones(4, 6)[::2, ::2], torch.ones(3, 2)),
        lambda: attention(*[torch.ones(1, 1, 2, 16) for _ in range(3)]),
        lambda: conv2d(torch.ones(1, 2, 4, 4), torch.ones(1, 2, 2, 2)),
        lambda: conv2d(torch.ones(1, 2, 1, 1), torch.ones(1, 2, 3, 3)),
        lambda: vector_add(torch.empty(0), torch.empty(0)),
        lambda: vector_add(torch.ones(2), torch.ones(2), backend="automatic"),
    ],
)
def test_invalid_inputs(call):
    with pytest.raises((ValueError, TypeError)):
        call()
