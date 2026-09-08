"""CPU tests of bounded HCCL evidence capture; no server or NPU required."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_flags, get_resources
from sglang.srt.utils import hccl_debug as debug
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHcclDebug(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def recorder(self, suffix="a"):
        recorder = debug.HcclDebugRecorder(
            Path(self.tmp.name) / suffix, rank=0, metadata={"world_size": 1}
        )
        (recorder.directory.parent / "START").touch()
        return recorder

    def events(self, recorder):
        return [json.loads(line) for line in recorder.path.read_text().splitlines()]

    @contextmanager
    def runtime_graph(self, decode="disabled", prefill="disabled", compile=False):
        # Install separate raw input and resolved namespace fixtures on the
        # actual context. Do not patch accessor bindings: that would miss the
        # regression where production reads a different source of config.
        graph = SimpleNamespace(
            decode=SimpleNamespace(backend=decode),
            prefill=SimpleNamespace(backend=prefill),
        )
        with (
            patch.object(
                get_context(),
                "_server_args",
                SimpleNamespace(cuda_graph_config=None, enable_torch_compile=True),
            ),
            patch.object(
                get_context(),
                "_config_bags",
                {
                    "exec": SimpleNamespace(
                        graph=SimpleNamespace(cuda_graph_config=graph)
                    )
                },
            ),
            get_flags().capture.override(enable_torch_compile=compile),
            get_resources().override(buffers={}),
            envs.SGLANG_DEBUG_HCCL_DIR.override(self.tmp.name),
        ):
            yield

    def test_resolved_eager_config_allows_raw_none(self):
        with self.runtime_graph():
            recorder = debug._get_recorder()
            self.assertIs(debug._get_recorder(), recorder)
            state = self.events(recorder)[0]["graph_state"]
            self.assertEqual(state["decode_backend"], "disabled")
            self.assertEqual(state["prefill_backend"], "disabled")
            self.assertFalse(state["enable_torch_compile"])

    def test_effective_graph_or_compile_still_rejected(self):
        for config in (
            {"decode": "full"},
            {"prefill": "tc_piecewise"},
            {"compile": True},
        ):
            with self.subTest(config=config), self.runtime_graph(**config):
                with self.assertRaisesRegex(RuntimeError, "Effective runtime values"):
                    debug._get_recorder()
                self.assertFalse(get_resources().buffers)

    def test_bfloat16_noncontiguous_nonfinite_and_empty(self):
        tensor = torch.tensor(
            [[1, float("nan")], [float("inf"), -2]], dtype=torch.bfloat16
        ).T
        cpu, stats = debug.tensor_summary(tensor)
        self.assertTrue(cpu.is_contiguous())
        self.assertEqual(stats["nan"], 1)
        self.assertEqual(stats["posinf"], 1)
        self.assertEqual(stats["min"], -2)
        self.assertEqual(
            stats["sha256"], debug.tensor_summary(tensor.contiguous())[1]["sha256"]
        )
        json.dumps(stats, allow_nan=False)
        self.assertEqual(debug.tensor_summary(torch.empty(0, 3))[1]["numel"], 0)
        self.assertEqual(debug.tensor_summary(torch.tensor(3))[1]["values"], 3)

    def test_disabled_preserves_original_callable(self):
        def operation(x):
            return x

        with envs.SGLANG_DEBUG_HCCL_DIR.override(""):
            self.assertIs(debug.hccl_trace("test")(operation), operation)

    def test_wait_for_start_does_not_consume_budget(self):
        recorder = self.recorder()
        trigger = recorder.directory.parent / "START"
        trigger.unlink()
        with envs.SGLANG_DEBUG_HCCL_DIR.override(self.tmp.name):

            @debug.hccl_trace("test", root=True, inputs=("x",))
            def operation(x):
                return x

        with patch.object(debug, "_get_recorder", return_value=recorder):
            operation(torch.ones(1))
            self.assertEqual(recorder.event_ct, 0)
            self.assertEqual(dict(recorder.steps), {})
            trigger.touch()
            operation(torch.ones(1))
        self.assertEqual(recorder.event_ct, 2)

    def test_inplace_before_after_and_dump_budget(self):
        with envs.SGLANG_DEBUG_HCCL_SAVE_TENSORS.override(True):
            recorder = self.recorder()
        recorder.max_dump_bytes = 4
        with envs.SGLANG_DEBUG_HCCL_DIR.override(self.tmp.name):

            @debug.hccl_trace("test", root=True, inputs=("x",), outputs=("result",))
            def operation(x):
                return x.add_(1)

        with patch.object(debug, "_get_recorder", return_value=recorder):
            x = torch.tensor([1.0])
            self.assertIs(operation(x), x)
        before, after = self.events(recorder)[1:]
        self.assertEqual(before["tensors"]["x"]["mean"], 1)
        self.assertEqual(after["tensors"]["result"]["mean"], 2)
        snapshot = torch.load(
            recorder.directory / before["tensors"]["x"]["file"], weights_only=True
        )
        self.assertEqual(snapshot.item(), 1)
        self.assertEqual(after["tensors"]["result"]["dump_skipped"], "byte_budget")
        self.assertEqual(recorder.stack, [])

    def test_layer_filter_skip_steps_and_exception_cleanup(self):
        recorder = self.recorder()
        recorder.layers = {"1"}
        recorder.skip_steps = 1
        recorder.max_steps = 1
        with envs.SGLANG_DEBUG_HCCL_DIR.override(self.tmp.name):

            @debug.hccl_trace("layer", layer=True, inputs=("x",))
            def layer(self, x):
                return x

            @debug.hccl_trace("root", root=True)
            def root():
                layer(SimpleNamespace(layer_idx=0), torch.ones(1))
                layer(SimpleNamespace(layer_idx=1), torch.ones(1))

        with patch.object(debug, "_get_recorder", return_value=recorder):
            root()
            self.assertEqual(recorder.event_ct, 0)
            root()
            count = recorder.event_ct
            root()
            self.assertEqual(recorder.event_ct, count)
        scopes = [e["scope"] for e in self.events(recorder)[1:]]
        self.assertIn(["root", "layer[1]"], scopes)
        self.assertNotIn(["root", "layer[0]"], scopes)
        recorder.steps.clear()
        recorder.skip_steps = 0
        with envs.SGLANG_DEBUG_HCCL_DIR.override(self.tmp.name):

            @debug.hccl_trace("fail", root=True)
            def fail():
                raise RuntimeError("test")

        with patch.object(debug, "_get_recorder", return_value=recorder):
            with self.assertRaises(RuntimeError):
                fail()
        self.assertEqual(recorder.stack, [])

    def test_event_limit_and_duplicate_run_rejected(self):
        recorder = self.recorder()
        recorder.max_events = 1
        recorder.stack.append({"root": ["test", 0], "name": "test"})
        recorder.emit("before", {}, (), {})
        recorder.emit("after", {}, (), {})
        self.assertEqual(
            [e["type"] for e in self.events(recorder)],
            ["manifest", "snapshot", "limit"],
        )
        with self.assertRaises(FileExistsError):
            self.recorder()

    def test_comparator_difference_and_missing_rank(self):
        spec = importlib.util.spec_from_file_location(
            "compare_hccl_debug",
            Path(__file__).resolve().parents[4] / "scripts/compare_hccl_debug.py",
        )
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        a, b = self.recorder("a"), self.recorder("b")
        for recorder, value in ((a, 1.0), (b, 2.0)):
            recorder.save_tensors = True
            recorder.stack.append({"root": ["test", 0], "name": "test"})
            recorder.emit("before", {"x": torch.tensor([value])}, ("x",), {})
            recorder.emit("after", {}, (), {})
        report, status = compare.compare_runs(
            a.directory.parent, b.directory.parent, tensors=True
        )
        self.assertEqual(status, 1)
        self.assertEqual(
            report["ranks"][0]["first_difference"]["numerical"]["max_abs"], 1
        )
        self.assertEqual(
            compare.compare_runs(a.directory.parent, a.directory.parent)[1], 0
        )
        records = self.events(b)
        records[0]["world_size"] = 2
        b.path.write_text("\n".join(json.dumps(e) for e in records) + "\n")
        self.assertEqual(
            compare.compare_runs(a.directory.parent, b.directory.parent)[1], 2
        )

    def test_allreduce_reference_and_alignment_details(self):
        spec = importlib.util.spec_from_file_location(
            "compare_hccl_debug",
            Path(__file__).resolve().parents[4] / "scripts/compare_hccl_debug.py",
        )
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        left, right = Path(self.tmp.name) / "left", Path(self.tmp.name) / "right"
        # FP64 sum = 1; BF16 serial rounding can lose the middle contribution.
        for directory, output in ((left, 1.0), (right, 0.0)):
            for rank, value in enumerate((256.0, 1.0, -256.0)):
                recorder = debug.HcclDebugRecorder(
                    directory, rank=rank, metadata={"world_size": 3}
                )
                recorder.save_tensors = True
                recorder.stack.append({"root": ["test", 0], "name": "tp.all_reduce"})
                metadata = {
                    "ranks": [0, 1, 2],
                    "world_size": 3,
                    "rank_in_group": rank,
                    "group": "tp0",
                }
                recorder.emit(
                    "before",
                    {"input_": torch.tensor([[value]], dtype=torch.bfloat16)},
                    ("input_",),
                    metadata,
                )
                recorder.emit(
                    "after",
                    {"result": torch.tensor([[output]], dtype=torch.bfloat16)},
                    ("result",),
                    metadata,
                )
        reference = compare.allreduce_reference(left, right, 1)
        self.assertTrue(reference["all_rank_inputs_match_between_modes"])
        self.assertTrue(
            reference["modes"]["right"]["outputs_bitwise_equal_across_group"]
        )
        for result in reference["modes"]["left"]["outputs"]:
            self.assertEqual(result["num_different_from_rounded_reference"], 0)
        for result in reference["modes"]["right"]["outputs"]:
            self.assertEqual(result["max_abs_vs_fp64"], 1)
        with self.assertRaisesRegex(ValueError, "not an AllReduce output"):
            compare.allreduce_reference(left, right, 0)
        path = right / "rank-1/events.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[1]["metadata"]["mode"] = "different"
        path.write_text("\n".join(json.dumps(e) for e in records) + "\n")
        report, status = compare.compare_runs(left, right)
        self.assertEqual(status, 2)
        self.assertEqual(
            report["ranks"][1]["alignment_mismatch"]["right_metadata"]["mode"],
            "different",
        )
        # Replacing a .pt without its JSON digest must fail closed.
        torch.save(
            torch.tensor([[2.0]], dtype=torch.bfloat16), right / "rank-1/000000-000.pt"
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            compare.allreduce_reference(left, right, 1)


if __name__ == "__main__":
    unittest.main()
