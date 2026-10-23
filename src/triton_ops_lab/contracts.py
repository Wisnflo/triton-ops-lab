"""Metadata guards for explicitly bound inference callables."""


def tensor_signature(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.requires_grad,
    )


class BoundTensorGuard:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)
        self.signatures = tuple(tensor_signature(t) for t in self.tensors)

    def check(self):
        if tuple(tensor_signature(t) for t in self.tensors) != self.signatures:
            raise RuntimeError("Bound tensor metadata/storage changed; prepare the callable again")
