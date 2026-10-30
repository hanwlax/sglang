"""CPU unit tests for the NPU CUDA-graph backend's Python orchestration."""

import unittest
from threading import Event, Thread
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=10, suite="base-a-test-npu")


def _make_backend(*, is_draft_worker=False):
    device_module = SimpleNamespace(
        current_device=mock.Mock(return_value=3),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
        Event=mock.Mock(),
    )
    runner = SimpleNamespace(
        device_module=device_module,
        enable_torch_compile=False,
        model_runner=SimpleNamespace(is_draft_worker=is_draft_worker),
    )
    with (
        get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
        mock.patch.object(mod.TorchMemorySaverAdapter, "create", return_value=None),
    ):
        return mod.NPUCudaGraphBackend(runner), runner


class TestNPUCudaGraphBackend(CustomTestCase):
    def test_replay_update_converts_legacy_sequence_lengths_to_int32_tensor(self):
        backend, runner = _make_backend()
        self.addCleanup(backend.cleanup)
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay_with_input_update(
            shape_key,
            seq_lens=[7, 9],
            attr_name="seq_lens",
            attr_type=torch.empty(0),
        )

        runner.device_module.set_device.assert_called_once_with(3)
        graph.replay.assert_called_once_with()
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(len(update_input), 1)
        self.assertEqual(set(update_input[0]), {"seq_lens"})
        self.assertEqual(update_input[0]["seq_lens"].dtype, torch.int32)
        self.assertTrue(torch.equal(update_input[0]["seq_lens"], torch.tensor([7, 9])))
        self.assertEqual(output, "output")

    def test_role_controls_preserve_replay_order_and_global_kill_switch(self):
        """Serial replay must not consume inputs while their update is blocked.

        Hold update on a real executor until the caller waits on its future.
        This deterministically distinguishes replay-before-wait from
        wait-before-replay, including opposite settings on the two roles.
        """
        cases = (
            # global, target, draft, expected target, expected draft
            (True, True, False, True, False),
            (True, False, True, False, True),
            (False, True, True, False, False),
            (True, True, True, True, True),
            (True, False, False, False, False),
        )
        for global_on, target_on, draft_on, target_expected, draft_expected in cases:
            for is_draft, expected in (
                (False, target_expected),
                (True, draft_expected),
            ):
                with (
                    self.subTest(
                        global_on=global_on,
                        target_on=target_on,
                        draft_on=draft_on,
                        is_draft=is_draft,
                    ),
                    envs.SGLANG_ENABLE_NPU_GRAPH_UPDATE_REPLAY_OVERLAP.override(
                        global_on
                    ),
                    envs.SGLANG_ENABLE_NPU_TARGET_GRAPH_UPDATE_REPLAY_OVERLAP.override(
                        target_on
                    ),
                    envs.SGLANG_ENABLE_NPU_DRAFT_GRAPH_UPDATE_REPLAY_OVERLAP.override(
                        draft_on
                    ),
                ):
                    self._check_update_replay_order(is_draft, expected)

    def _check_update_replay_order(self, is_draft, expected_overlap):
        backend, runner = _make_backend(is_draft_worker=is_draft)
        update_started = Event()
        release_update = Event()
        update_done = Event()
        result_entered = Event()
        replayed = Event()
        previous_replay_done = Event()
        backend._rebind_fence = SimpleNamespace(synchronize=previous_replay_done.set)
        inputs = [{"actual_seq_lengths_kv": [7, 9]}]
        observed_update_inputs = []
        replay_saw_update_done = []
        outputs = []
        errors = []
        shape_key = ShapeKey(size=2)

        def update(*, cpu_update_input):
            if not previous_replay_done.is_set():
                raise AssertionError("update started before the preceding replay fence")
            observed_update_inputs.append(cpu_update_input)
            update_started.set()
            if not release_update.wait(5):
                raise TimeoutError("test did not release the graph update")
            update_done.set()

        def replay():
            replay_saw_update_done.append(update_done.is_set())
            replayed.set()

        backend._graphs[shape_key] = SimpleNamespace(update=update, replay=replay)
        backend._outputs[shape_key] = "output"
        real_submit = backend._update_executor.submit

        def submit(*args, **kwargs):
            future = real_submit(*args, **kwargs)
            real_result = future.result

            def result():
                result_entered.set()
                return real_result(timeout=5)

            future.result = result
            return future

        def run():
            try:
                outputs.append(
                    backend.replay_with_input_update(
                        shape_key, seq_lens=None, cpu_update_input=inputs
                    )
                )
            except BaseException as error:
                errors.append(error)

        caller = Thread(target=run)
        try:
            with mock.patch.object(
                backend._update_executor, "submit", side_effect=submit
            ):
                caller.start()
                self.assertTrue(update_started.wait(5))
                self.assertTrue(result_entered.wait(5))
                self.assertEqual(replayed.is_set(), expected_overlap)
                release_update.set()
                caller.join(5)
                self.assertFalse(caller.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(outputs, ["output"])
            self.assertEqual(replay_saw_update_done, [not expected_overlap])
            self.assertIs(observed_update_inputs[0], inputs)
            self.assertIs(
                backend._rebind_fence, runner.device_module.Event.return_value
            )
        finally:
            release_update.set()
            caller.join(5)
            backend.cleanup()


if __name__ == "__main__":
    unittest.main()
