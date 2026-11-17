import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import torch
from view_matrix import Matrices, handler_for, heatmap


def q(**kw):
    return {k: [str(v)] for k, v in kw.items()}


class ViewerChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "full_L0_layer.output_TP0.pt"
        self.x = torch.arange(35).reshape(5, 7).to(torch.bfloat16) / 16
        torch.save({"tensor": self.x, "rows": [10, 11, 12, 13, 14]}, self.path)
        self.m = Matrices(root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_window_coordinates_and_clipped_edge(self):
        r = self.m.window(q(id=0, row=3, col=5, rows=8, cols=8, view="table"))
        self.assertEqual(r["values"], self.x[3:, 5:].tolist())
        self.assertEqual(r["window"], dict(row=3, col=5, rows=2, cols=2))
        self.assertEqual(r["metadata"]["rows"], [10, 11, 12, 13, 14])
        self.assertEqual(self.m.cell(q(id=0, row=3, col=5))["token_row"], 13)

    def test_signed_max_and_padding(self):
        x = torch.tensor(
            [
                [1.0, -9.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, -8.0, 6.0],
                [1.0, 1.0, 1.0, 1.0, 7.0],
            ]
        )
        h = heatmap(x, max_rows=2, max_cols=3)
        self.assertEqual((h["row_step"], h["col_step"]), (2, 2))
        self.assertEqual(h["values"], [[-9.0, -8.0, 6.0], [1.0, 1.0, 7.0]])
        self.assertEqual(h["flags"], [[0, 0, 0], [0, 0, 0]])

    def test_nonfinite_flags_and_json(self):
        x = torch.tensor([[float("nan"), float("inf"), float("-inf"), -2.0]])
        h = heatmap(x, max_rows=1, max_cols=1)
        self.assertEqual(h["flags"], [[7]])
        self.assertEqual(h["values"], [[-2.0]])
        self.assertEqual(
            heatmap(torch.full((2, 2), float("nan")))["values"],
            [[None, None], [None, None]],
        )
        torch.save(x, self.path)
        result = self.m.window(q(id=0, view="table"))
        self.assertEqual(result["values"][0], ["NaN", "+Inf", "-Inf", -2.0])
        json.dumps(result, allow_nan=False)

    def test_integer_precision(self):
        torch.save(torch.tensor([[2**63 - 1, -(2**63)]], dtype=torch.int64), self.path)
        self.assertEqual(
            self.m.window(q(id=0, view="table"))["values"],
            [[str(2**63 - 1), str(-(2**63))]],
        )

    def test_reject_bad_input(self):
        for params in [
            q(id="../private.pt"),
            q(id=0, row=-1),
            q(id=0, col=7),
            q(id=0, rows=0),
            q(id=0, view="foo"),
        ]:
            with self.assertRaises(ValueError):
                self.m.window(params)
        torch.save(torch.ones(2, 3, 4), self.path)
        with self.assertRaises(ValueError):
            Matrices(root=self.root).load("0")

    def test_production_shape_extreme(self):
        x = torch.zeros(852, 8192, dtype=torch.bfloat16)
        x[851, 8191] = -16
        torch.save({"tensor": x, "rows": list(range(852))}, self.path)
        h = self.m.window(q(id=0, row=0, col=0, rows=852, cols=8192))
        self.assertEqual(h["stats"]["elements"], 852 * 8192)
        self.assertEqual(h["values"][-1][-1], -16.0)
        self.assertTrue(h["aggregated"])
        with self.assertRaises(ValueError):
            self.m.window(q(id=0, rows=852, cols=8192, view="table"))

    def test_logits_row_and_single_file(self):
        path = self.root / "full_L-1_lm_head.logits_TP7.pt"
        torch.save({"tensor": torch.zeros(1, 10), "rows": [851]}, path)
        m = Matrices(file=path)
        self.assertEqual(m.entries[0]["tp"], 7)
        self.assertEqual(m.cell(q(id=0, row=0, col=0))["token_row"], 851)

    def test_http(self):
        server = HTTPServer(("127.0.0.1", 0), handler_for(self.m))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/api/window?id=0&view=table") as f:
                self.assertEqual(json.load(f)["values"], self.x.tolist())
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(base + "/api/meta?id=../private.pt")
            self.assertEqual(cm.exception.code, 400)
            with urllib.request.urlopen(base) as f:
                self.assertIn("矩阵查看器", f.read().decode())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
