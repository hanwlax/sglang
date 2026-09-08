"""CPU tests of bounded HCCL evidence capture; no server or NPU required."""

import importlib.util
import json
import sys
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

    def replay_module(self):
        scripts = Path(__file__).resolve().parents[4] / "scripts"
        with patch.object(sys, "path", [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "replay_hccl_allreduce", scripts / "replay_hccl_allreduce.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

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
        with self.runtime_graph(), envs.SGLANG_SIMULATE_ACC_LEN.override(1.0):
            recorder = debug._get_recorder()
            self.assertIs(debug._get_recorder(), recorder)
            state = self.events(recorder)[0]["graph_state"]
            self.assertEqual(state["decode_backend"], "disabled")
            self.assertEqual(state["prefill_backend"], "disabled")
            self.assertFalse(state["enable_torch_compile"])
            self.assertEqual(
                self.events(recorder)[0]["simulation"]["SGLANG_SIMULATE_ACC_LEN"], 1.0
            )

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
                    directory,
                    rank=rank,
                    metadata={"world_size": 3, "hostname": "fixture-host"},
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
        report, _ = compare.compare_runs(left, right)
        report["allreduce_reference"] = reference
        brief = compare.brief_report(report)
        self.assertEqual(brief["rank_groups"][0]["ranks"], [0, 1, 2])
        self.assertEqual(
            len(brief["allreduce_reference"]["modes"]["left"]["outputs"]), 1
        )
        self.assertEqual(len(reference["modes"]["left"]["outputs"]), 3)
        replay = self.replay_module()
        source, golden, _, hashes = replay.prepare_case(left, right, 1, rank=1)
        self.assertEqual(source.item(), 1)
        self.assertEqual(golden.item(), 1)
        self.assertEqual(len(hashes), 3)
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
        with self.assertRaisesRegex(ValueError, "does not match"):
            replay.prepare_case(left, right, 1)

    def test_replay_restores_input_and_synchronizes_every_iteration(self):
        replay = self.replay_module()
        source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        steps, outputs = [], []

        def reduce(value):
            self.assertTrue(torch.equal(value, source))
            steps.append("reduce")
            value.mul_(3)

        replay.replay_iterations(
            source,
            3,
            2,
            allocate=torch.empty_like,
            synchronize=lambda: steps.append("sync"),
            reduce=reduce,
            observe=lambda i, phase, output: outputs.append((i, phase, output)),
        )
        self.assertEqual(steps, ["sync", "reduce", "sync"] * 5)
        self.assertEqual(
            [phase for _, phase, _ in outputs], ["warmup"] * 2 + ["measured"] * 3
        )
        self.assertTrue(torch.equal(source, torch.tensor([1, 2], dtype=torch.bfloat16)))
        self.assertTrue(all(torch.equal(value, source * 3) for _, _, value in outputs))
        self.assertEqual(
            replay.error_metrics(source, source.double())["relative_l2_vs_fp64"], 0
        )
        self.assertEqual(
            replay.error_metrics(torch.tensor([float("nan")]), torch.ones(1))[
                "nonfinite"
            ],
            1,
        )

    def test_replay_summary_rejects_incomplete_or_tampered_results(self):
        replay = self.replay_module()
        root = Path(self.tmp.name)
        for rank in range(2):
            directory = root / f"rank-{rank}"
            directory.mkdir()
            value = torch.tensor([3.0], dtype=torch.bfloat16)
            torch.save(value, directory / "output.pt")
            entry = {
                "iteration": 0,
                "sha256": replay.digest(value),
                "file": "output.pt",
                "matches_model": {"left": True, "right": False},
                "nonfinite": 0,
            }
            report = {
                "rank": rank,
                "world_size": 2,
                "requested_mode": "AIV",
                "source_event": 3,
                "source_root": ["test", 0],
                "source_scope": ["tp.all_reduce"],
                "group_ranks": [0, 1],
                "all_input_sha256": ["a", "b"],
                "shape": [1],
                "dtype": "torch.bfloat16",
                "warmup": 0,
                "iterations": 1,
                "records": [entry],
                "completed": True,
            }
            (directory / "report.json").write_text(json.dumps(report))
        summary = replay.summarize(root)
        self.assertTrue(summary["all_ranks_repeatable_including_warmup"])
        self.assertTrue(summary["all_outputs_match_model"]["left"])
        self.assertTrue(summary["all_ranks_equal_each_iteration"])
        (root / "rank-1/report.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            replay.summarize(root)
        (root / "rank-1/report.json").write_text(json.dumps(report))
        torch.save(torch.tensor([4.0], dtype=torch.bfloat16), root / "rank-1/output.pt")
        with self.assertRaisesRegex(ValueError, "does not match"):
            replay.summarize(root)

    def test_scope_inspection_survives_alignment_stop(self):
        spec = importlib.util.spec_from_file_location(
            "compare_hccl_debug",
            Path(__file__).resolve().parents[4] / "scripts/compare_hccl_debug.py",
        )
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        a, b = self.recorder("inspect-a"), self.recorder("inspect-b")
        for recorder, count in ((a, 1), (b, 2)):
            recorder.stack.append({"root": ["decode", 0], "name": "dspark.accept"})
            recorder.emit("before", {}, (), {"count": count})
            recorder.emit("after", {"result": torch.tensor([count])}, ("result",), {})
        self.assertEqual(
            compare.compare_runs(a.directory.parent, b.directory.parent)[1], 2
        )
        inspection = compare.inspect_scopes(
            b.directory.parent, ["dspark.accept"], limit=2
        )
        self.assertEqual(inspection["events"][1]["tensors"]["result"]["values"], [2])
        self.assertFalse(
            inspection["events"][1]["tensors"]["result"]["saved_file_exists"]
        )
        self.assertIn("unknown", inspection["accept_length_evidence"])
        self.assertEqual(
            compare.inspect_scopes(b.directory.parent, ["missing"])["matching_events"],
            0,
        )
        self.assertEqual(
            compare.inspect_scopes(b.directory.parent, ["dspark.accept"], limit=1)[
                "omitted_events"
            ],
            1,
        )
        records = self.events(b)
        records[0]["simulation"] = {"SGLANG_SIMULATE_ACC_LEN": 1.0}
        b.path.write_text("\n".join(json.dumps(e) for e in records) + "\n")
        self.assertIn(
            "simulated",
            compare.inspect_scopes(b.directory.parent, ["dspark.accept"])[
                "accept_length_evidence"
            ],
        )


if __name__ == "__main__":
    unittest.main()
