"""CPU-only numerical/metadata checks for the offline LongCat boundary tool.

Run: python3 -m pytest -q test/manual/ascend/test_longcat_attn0_moe_boundary.py
No model, NPU, SDK or production SGLang imports are needed.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "longcat2_950dt"
# The installed repository path is the supported invocation.
sys.path.insert(0, str(SCRIPTS))
import check_attn0_moe_boundary as boundary


def test_direct_bf16_rounding_at_midpoints():
    midpoint = 1 + 1 / 256
    x = torch.tensor(
        [
            midpoint - 1e-10,
            midpoint,
            midpoint + 1e-10,
            -midpoint + 1e-10,
            -midpoint,
            -midpoint - 1e-10,
        ],
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [1, 1, 1 + 1 / 128, -1, -1, -(1 + 1 / 128)], dtype=torch.bfloat16
    )
    assert torch.equal(boundary.round_bf16(x), expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("fp64_once", 1),
        ("bf16_serial_0_to_7", 0),
        ("bf16_serial_7_to_0", 1),
        ("bf16_pairwise", 0),
    ],
)
def test_reduction_cancellation_distinguishes_orders(method, expected):
    parts = [
        torch.tensor([[v]], dtype=torch.bfloat16) for v in [256, 1, -256, 0, 0, 0, 0, 0]
    ]
    assert boundary.sum_ranks(parts, method).item() == expected


def test_fused_norm_cannot_be_replaced_by_norm_of_rounded_residual():
    unrounded = torch.tensor([[257 / 256, 1]], dtype=torch.float64)
    weight = torch.ones(2, dtype=torch.bfloat16)
    fused = boundary.rms(unrounded, weight, 1e-5, torch.float64)
    staged = boundary.rms(boundary.round_bf16(unrounded), weight, 1e-5, torch.float64)
    assert torch.equal(fused, torch.tensor([[1, 0.99609375]], dtype=torch.bfloat16))
    assert torch.equal(staged, torch.ones((1, 2), dtype=torch.bfloat16))


def test_metrics_full_coordinates_and_nonfinite_accounting():
    actual = torch.zeros((3, 11), dtype=torch.bfloat16)
    reference = actual.clone()
    actual[2, 10] = 0.125
    actual[1, 9] = float("nan")
    r = boundary.difference(actual, reference)
    assert r["elements"] == 33 and r["different"] == 2
    assert r["max_abs"] == 0.125 and r["max_difference"]["coordinate"] == [2, 10]
    assert r["max_difference"]["actual"] == 0.125
    assert r["first_nonfinite"]["coordinate"] == [1, 9]
    assert r["nonfinite_actual"] == 1 and r["nonfinite_reference"] == 0
    assert sum(r["bins"].values()) + r["nonfinite_pairs"] == 33


def test_exported_params_hash_rejects_changed_values():
    w = torch.ones(8192, dtype=torch.bfloat16)
    params = {
        "schema": 1,
        "kind": "longcat_l0_post_attention_norm",
        "layer": 0,
        "hidden_size": 8192,
        "weight_key": boundary.WEIGHT_KEY,
        "assumed_runtime_dtype": "torch.bfloat16",
        "epsilon": 1e-5,
        "weight_bf16_sha256": boundary.weight_hash(w),
        "weight": w.float().tolist(),
    }
    assert torch.equal(boundary.decode_params(params)[0], w)
    params["weight"][844] = 0.5
    with pytest.raises(ValueError, match="hash"):
        boundary.decode_params(params)


def test_capture_reader_rejects_other_run_and_missing_rank(tmp_path):
    folder = tmp_path / "full"
    folder.mkdir()
    route = {
        "identity": {"input_hash": "test", "input_ids": [1]},
        "mode": "ids_weights",
    }
    payload = {
        "schema": 1,
        "layer": 0,
        "point": "moe.input",
        "tp": 0,
        "identity": route["identity"],
        "route_mode": "ids_weights",
        "mode": "EXTEND",
        "rows": list(range(852)),
        "tensor": torch.zeros((852, 8192), dtype=torch.bfloat16),
    }
    path = folder / "full_L0_moe.input_TP0.pt"
    torch.save(payload, path)
    assert tuple(boundary.read_capture(tmp_path, "moe", 0, route).shape) == (852, 8192)
    bad_route = copy.deepcopy(route)
    bad_route["identity"]["input_hash"] = "different"
    with pytest.raises(ValueError, match="mixed capture"):
        boundary.read_capture(tmp_path, "moe", 0, bad_route)
    with pytest.raises(FileNotFoundError):
        boundary.read_capture(tmp_path, "moe", 7, route)
    payload["rows"] = list(reversed(payload["rows"]))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="mixed capture"):
        boundary.read_capture(tmp_path, "moe", 0, route)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
