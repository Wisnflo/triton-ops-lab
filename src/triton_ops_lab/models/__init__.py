"""Small complete forward modules with identical weights across backends."""

import torch
import torch.nn.functional as F

from ..ops import attention, conv2d, matmul


class TinyTransformer(torch.nn.Module):
    def __init__(self, width=64, heads=2):
        super().__init__()
        if width % heads or width // heads not in {32, 64, 128}:
            raise ValueError("head dimension must be 32/64/128")
        self.width, self.heads = width, heads
        for name, shape in {
            "qkv": (width, 3 * width),
            "proj": (width, width),
            "up": (width, 2 * width),
            "down": (2 * width, width),
        }.items():
            self.register_buffer(name, torch.randn(shape) / width**0.5)
        self.register_buffer("bias", torch.randn(2 * width) * 0.01)

    def make_plan(self, shape, dtype, policy):
        b, n, d = shape
        if d != self.width:
            raise ValueError("Input width does not match model")
        specifications = {
            "qkv": dict(op="matmul", shape=[b * n, 3 * d, d]),
            "attention": dict(
                op="attention", shape=[b, self.heads, n, d // self.heads], causal=True
            ),
            "proj": dict(op="matmul", shape=[b * n, d, d]),
            "up": dict(op="matmul", shape=[b * n, 2 * d, d], bias=True, activation="gelu"),
            "down": dict(op="matmul", shape=[b * n, d, 2 * d]),
        }
        return dict(
            input_shape=list(shape),
            dtype=dtype,
            layers={
                name: dict(
                    case={**case, "dtype": dtype}, config=policy.choose({**case, "dtype": dtype})
                )
                for name, case in specifications.items()
            },
        )

    @torch.no_grad()
    def forward(self, x, backend="torch", plan=None):
        if plan is not None and (
            backend != "triton"
            or list(x.shape) != plan["input_shape"]
            or str(x.dtype).removeprefix("torch.") != plan["dtype"]
        ):
            raise ValueError("A Triton policy plan requires its original input shape and dtype")

        def tile(name):
            return None if plan is None else plan["layers"][name]["config"]

        b, n, d = x.shape
        z = F.layer_norm(x, (d,))
        qkv = matmul(z.reshape(-1, d), self.qkv, backend=backend, config=tile("qkv")).reshape(
            b, n, 3, self.heads, d // self.heads
        )
        q, k, v = [qkv[:, :, i].transpose(1, 2).contiguous() for i in range(3)]
        a = (
            attention(q, k, v, causal=True, backend=backend, config=tile("attention"))
            .transpose(1, 2)
            .contiguous()
        )
        x = x + matmul(a.reshape(-1, d), self.proj, backend=backend, config=tile("proj")).reshape(
            b, n, d
        )
        z = F.layer_norm(x, (d,)).reshape(-1, d)
        z = matmul(
            z,
            self.up,
            self.bias,
            activation="gelu",
            backend=backend,
            variant="fused",
            config=tile("up"),
        )
        return x + matmul(z, self.down, backend=backend, config=tile("down")).reshape(b, n, d)


class TinyCNN(torch.nn.Module):
    def __init__(self, channels=3, hidden=8):
        super().__init__()
        self.register_buffer("w1", torch.randn(hidden, channels, 3, 3) / (channels * 9) ** 0.5)
        self.register_buffer("w2", torch.randn(hidden, hidden, 1, 1) / hidden**0.5)

    def make_plan(self, shape, dtype, policy):
        b, c, h, w = shape
        if c != self.w1.shape[1]:
            raise ValueError("Input channels do not match model")
        hidden = self.w1.shape[0]
        specifications = {
            "conv1": dict(op="conv2d", shape=[b, c, h, w, hidden, 3], stride=1, padding=1),
            "conv2": dict(op="conv2d", shape=[b, hidden, h, w, hidden, 1], stride=2, padding=0),
        }
        return dict(
            input_shape=list(shape),
            dtype=dtype,
            layers={
                name: dict(
                    case={**case, "dtype": dtype}, config=policy.choose({**case, "dtype": dtype})
                )
                for name, case in specifications.items()
            },
        )

    @torch.no_grad()
    def forward(self, x, backend="torch", plan=None):
        if plan is not None and (
            backend != "triton"
            or list(x.shape) != plan["input_shape"]
            or str(x.dtype).removeprefix("torch.") != plan["dtype"]
        ):
            raise ValueError("A Triton policy plan requires its original input shape and dtype")

        def tile(name):
            return None if plan is None else plan["layers"][name]["config"]

        z = torch.relu(conv2d(x, self.w1, padding=1, backend=backend, config=tile("conv1")))
        z = torch.relu(conv2d(z, self.w2, stride=2, backend=backend, config=tile("conv2")))
        return z.mean(dim=(-1, -2))
