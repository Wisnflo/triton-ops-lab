"""A strict, audited compile baseline. Pinned to the project's PyTorch version."""

import torch


class CompileAudit:
    def __init__(self, mm_compat=False):
        self.graphs = []
        self.mm_compat = mm_compat

    def backend(self, gm, example_inputs, **kwargs):
        from torch._inductor import compile as inductor_compile

        compiled = inductor_compile(gm, example_inputs, **kwargs)
        self.graphs.append(
            {
                "code": gm.code,
                "targets": [str(n.target) for n in gm.graph.nodes if n.op == "call_function"],
            }
        )
        return compiled

    def snapshot(self):
        return {
            "fullgraph": True,
            "dynamic": False,
            "cuda_graphs_in_inductor": False,
            "cache_reset_before_case": True,
            "graph_count": len(self.graphs),
            "mm_policy": "explicit_external_fp32_mm" if self.mm_compat else "native_aten",
            "opaque_external_ops": [
                target
                for graph in self.graphs
                for target in graph["targets"]
                if "triton_ops_lab.mm_fp32" in target
            ],
            "graphs": self.graphs,
        }

    def require_single_graph(self):
        if len(self.graphs) != 1:
            raise RuntimeError(f"Compile audit expected exactly one graph, got {len(self.graphs)}")


def compile_strict(fn, mm_compat=False):
    # invoke's code object is shared by all cases. Clear Dynamo's process cache
    # before each new case so shapes/operators cannot exhaust a shared limit.
    # Disk/Inductor caches may still be warm; first_call_ms is not cold compilation.
    torch.compiler.reset()
    audit = CompileAudit(mm_compat)
    compiled = torch.compile(
        fn,
        backend=audit.backend,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False},
    )
    compiled.compile_audit = audit
    return compiled
