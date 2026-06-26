import os
from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.fx as fx

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compile_phase import (
    get_pcg_capture_stream,
    is_in_torch_compile_warmup,
)
from sglang.srt.compilation.cuda_piecewise_backend import (
    CUDAPiecewiseBackend,
    weak_ref_tensors,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    get_tc_piecewise_forward_context,
)

# SGLANG_NPU_PROFILE_FWD=N  — profile forward #N with torch_npu.profiler
# SGLANG_NPU_PROFILE_DIR=path — output directory (default: /tmp/npu_profile)
_PROFILE_FWD = int(os.environ.get("SGLANG_NPU_PROFILE_FWD", "0"))
_PROFILE_DIR = os.environ.get("SGLANG_NPU_PROFILE_DIR", "/tmp/npu_profile")
_profile_ctx = None
_pcg_fwd_count: int = 0


class NPUPiecewiseBackend(CUDAPiecewiseBackend):
    def __init__(
        self,
        graph: fx.GraphModule,
        compile_config: CompilationConfig,
        inductor_config: dict[str, Any],
        graph_pool: Any,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        compiled_graph_for_general_shape: Callable,
        sglang_backend,
    ):
        super().__init__(
            graph,
            compile_config,
            inductor_config,
            graph_pool,
            piecewise_compile_index,
            total_piecewise_compiles,
            sym_shape_indices,
            compiled_graph_for_general_shape,
            sglang_backend,
        )

    def __call__(self, *args):
        if is_in_torch_compile_warmup():
            return self.compiled_graph_for_general_shape(*args)

        if len(self.sym_shape_indices) == 0:
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]
        if runtime_shape not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[runtime_shape]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.cudagraph is None:
            if entry.num_finished_warmup < 1:  # noqa
                entry.num_finished_warmup += 1
                return entry.runnable(*args)

            if self.compile_config.get_enable_debug_mode():
                input_addresses = [
                    x.data_ptr() for x in args if isinstance(x, torch.Tensor)
                ]
                entry.input_addresses = input_addresses
            npugraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.npu.graph(
                    npugraph,
                    pool=self.graph_pool,
                    stream=get_pcg_capture_stream(),
                    auto_dispatch_capture=True,
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = npugraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.compile_config.get_enable_debug_mode():
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        if _PROFILE_FWD:
            global _profile_ctx, _pcg_fwd_count
            if self.is_first_graph:
                _pcg_fwd_count += 1
                if _pcg_fwd_count == _PROFILE_FWD:
                    import torch_npu.profiler as npu_prof

                    chunk_tag = ""
                    ctx = get_tc_piecewise_forward_context()
                    if ctx and ctx.forward_batch is not None:
                        fb = ctx.forward_batch
                        prefix = int(fb.extend_prefix_lens[0].item()) if fb.extend_prefix_lens is not None and len(fb.extend_prefix_lens) > 0 else -1
                        seq_kv = int(fb.seq_lens[0].item()) if fb.seq_lens is not None and len(fb.seq_lens) > 0 else -1
                        chunk_tag = f"_prefix{prefix}_seqkv{seq_kv}"

                    out_dir = os.path.join(
                        _PROFILE_DIR, f"pcg_fwd{_pcg_fwd_count}_shape{runtime_shape}{chunk_tag}"
                    )
                    os.makedirs(out_dir, exist_ok=True)
                    _profile_ctx = npu_prof.profile(
                        activities=[
                            npu_prof.ProfilerActivity.CPU,
                            npu_prof.ProfilerActivity.NPU,
                        ],
                        on_trace_ready=npu_prof.tensorboard_trace_handler(out_dir),
                    )
                    _profile_ctx.__enter__()
                    print(
                        f"[PCG Profile] START fwd#{_pcg_fwd_count} "
                        f"shape={runtime_shape}{chunk_tag} -> {out_dir}"
                    )

        entry.cudagraph.replay()

        if _PROFILE_FWD and self.is_last_graph and _profile_ctx is not None:
            _profile_ctx.__exit__(None, None, None)
            print(f"[PCG Profile] END fwd#{_pcg_fwd_count}")
            _profile_ctx = None

        return entry.output
