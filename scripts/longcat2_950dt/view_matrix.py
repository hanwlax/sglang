#!/usr/bin/env python3
"""Offline, CPU-only viewer for LongCat two-dimensional activation dumps."""

import argparse
import json
import math
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch

PATTERN = re.compile(r"full_L(-?\d+)_(.+)_TP(\d+)\.pt")
MAX_TABLE = 64


def number(value):
    value = value.item() if isinstance(value, torch.Tensor) else value
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)  # Preserve int64 values across JSON / JavaScript.
    return value


def integer(query, name, default=None):
    value = query.get(name, [default])[0]
    if value is None:
        raise ValueError(f"缺少参数 {name}")
    return int(value)


def summarize(x):
    finite = torch.isfinite(x)
    values = x[finite]
    return {
        "elements": x.numel(),
        "finite": int(finite.sum()),
        "nan": int(torch.isnan(x).sum()),
        "posinf": int(torch.isposinf(x).sum()),
        "neginf": int(torch.isneginf(x).sum()),
        "zeros": int((x == 0).sum()),
        "min": number(values.min()) if values.numel() else None,
        "max": number(values.max()) if values.numel() else None,
    }


def heatmap(x, max_rows=256, max_cols=512):
    """Keep the signed largest finite magnitude per tile; track nonfinites separately."""
    height, width = x.shape
    rs = math.ceil(height / max_rows)
    cs = math.ceil(width / max_cols)
    nr, nc = math.ceil(height / rs), math.ceil(width / cs)
    padded = torch.full((nr * rs, nc * cs), float("nan"), dtype=torch.float64)
    padded[:height, :width] = x
    tiles = padded.reshape(nr, rs, nc, cs).permute(0, 2, 1, 3).reshape(nr, nc, -1)
    finite = torch.isfinite(tiles)
    indices = tiles.abs().masked_fill(~finite, -1).argmax(-1, keepdim=True)
    representatives = tiles.gather(-1, indices).squeeze(-1)
    valid = finite.any(-1)
    values = [
        [value if ok else None for value, ok in zip(row, mask)]
        for row, mask in zip(representatives.tolist(), valid.tolist())
    ]
    # Padding must not be reported as a captured NaN.
    flags = torch.zeros((nr * rs, nc * cs), dtype=torch.int32)
    flags[:height, :width] = (
        torch.isnan(x).int() + 2 * torch.isposinf(x).int() + 4 * torch.isneginf(x).int()
    )
    flags = flags.reshape(nr, rs, nc, cs).permute(0, 2, 1, 3).reshape(nr, nc, -1)
    status = sum(((flags & bit) != 0).any(-1).int() * bit for bit in (1, 2, 4))
    return {
        "values": values,
        "flags": status.tolist(),
        "row_step": rs,
        "col_step": cs,
        "aggregated": rs != 1 or cs != 1,
        "method": "signed_max_abs_finite",
    }


class Matrices:
    def __init__(self, root=None, file=None, label="本地矩阵"):
        self.label = label
        if file:
            paths = [Path(file).expanduser().resolve(strict=True)]
        else:
            directory = Path(root).expanduser().resolve(strict=True)
            if (directory / "full").is_dir():
                directory /= "full"
            paths = sorted(directory.glob("full_*.pt"))
        self.entries = []
        self.paths = {}
        for path in paths:
            match = PATTERN.fullmatch(path.name)
            if not match and not file:
                continue
            key = str(len(self.entries))
            self.paths[key] = path
            self.entries.append(
                {
                    "id": key,
                    "file": path.name,
                    "layer": int(match[1]) if match else -1,
                    "point": match[2] if match else path.stem,
                    "tp": int(match[3]) if match else 0,
                }
            )
        if not self.entries:
            raise ValueError(
                "没有找到 full_L*_TP*.pt；请指定单轮采集目录、full/ 或 --file"
            )
        self.cached = None

    def load(self, key):
        if key not in self.paths:
            raise ValueError("未知矩阵 ID")
        if self.cached is not None and self.cached[0] == key:
            return self.cached[1:]
        # Keep only one matrix resident. Never invoke the unrestricted pickle loader.
        self.cached = None
        payload = torch.load(self.paths[key], map_location="cpu", weights_only=True)
        if isinstance(payload, torch.Tensor):
            tensor, payload = payload, {}
        elif isinstance(payload, dict) and isinstance(
            payload.get("tensor"), torch.Tensor
        ):
            tensor = payload["tensor"]
        else:
            raise ValueError("需要二维 Tensor，或包含 tensor 字段的采集字典")
        if tensor.ndim != 2 or not all(tensor.shape) or tensor.is_complex():
            raise ValueError(f"仅支持非空二维实数矩阵，实际 shape={list(tensor.shape)}")
        if tensor.layout != torch.strided:
            raise ValueError("仅支持普通稠密矩阵")
        tensor = tensor.detach().contiguous()
        rows = payload.get("rows", list(range(tensor.shape[0])))
        if len(rows) != tensor.shape[0] or any(type(r) is not int for r in rows):
            raise ValueError("采集 rows 元数据与张量行数不一致")
        identity = payload.get("identity", {})
        metadata = {
            **self.entries[int(key)],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "rows": rows,
            "input_hash": identity.get("input_hash"),
            "mode": payload.get("mode"),
            "route_mode": payload.get("route_mode"),
        }
        self.cached = (key, tensor, metadata)
        return tensor, metadata

    def window(self, query):
        tensor, metadata = self.load(query.get("id", [""])[0])
        r, c = integer(query, "row", 0), integer(query, "col", 0)
        nr, nc = integer(query, "rows", 16), integer(query, "cols", 12)
        if not (0 <= r < tensor.shape[0] and 0 <= c < tensor.shape[1]):
            raise ValueError("起始行列超出矩阵范围；行列编号从 0 开始")
        if nr <= 0 or nc <= 0:
            raise ValueError("行数、列数必须为正整数")
        endr, endc = min(r + nr, tensor.shape[0]), min(c + nc, tensor.shape[1])
        mode = query.get("view", ["heatmap"])[0]
        if mode not in ("heatmap", "table"):
            raise ValueError("未知显示方式")
        if mode == "table" and (endr - r > MAX_TABLE or endc - c > MAX_TABLE):
            raise ValueError("数字表格最多显示 64×64；请缩小窗口或点击“局部 16×12”")
        original = tensor[r:endr, c:endc]
        x = original.double()
        result = {
            "metadata": metadata,
            "window": {"row": r, "col": c, "rows": endr - r, "cols": endc - c},
            "stats": summarize(x),
        }
        if mode == "table":
            result["values"] = [[number(v) for v in line] for line in original.tolist()]
        else:
            result.update(heatmap(x))
        return result

    def cell(self, query):
        tensor, meta = self.load(query.get("id", [""])[0])
        r, c = integer(query, "row"), integer(query, "col")
        if not (0 <= r < tensor.shape[0] and 0 <= c < tensor.shape[1]):
            raise ValueError("单元格坐标超出范围")
        return {
            "row": r,
            "token_row": meta["rows"][r],
            "col": c,
            "value": number(tensor[r, c]),
        }


def handler_for(matrices):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    data = Path(__file__).with_name("matrix_viewer.html").read_bytes()
                    mime = "text/html; charset=utf-8"
                else:
                    if parsed.path == "/api/catalog":
                        result = {"label": matrices.label, "entries": matrices.entries}
                    elif parsed.path == "/api/meta":
                        _, result = matrices.load(query.get("id", [""])[0])
                    elif parsed.path == "/api/window":
                        result = matrices.window(query)
                    elif parsed.path == "/api/cell":
                        result = matrices.cell(query)
                    else:
                        self.send_error(404)
                        return
                    data = json.dumps(
                        result, ensure_ascii=False, allow_nan=False
                    ).encode()
                    mime = "application/json; charset=utf-8"
                self.respond(200, data, mime)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                self.respond(
                    400,
                    json.dumps({"error": str(error)}, ensure_ascii=False).encode(),
                    "application/json",
                )
            except Exception as error:
                self.respond(
                    500,
                    json.dumps(
                        {"error": f"读取失败：{type(error).__name__}: {error}"},
                        ensure_ascii=False,
                    ).encode(),
                    "application/json",
                )

        def respond(self, status, data, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt, *args):
            pass  # Keep the local terminal quiet; API errors appear in the page.

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="单轮采集目录或其中的 full/ 目录")
    source.add_argument("--file", help="单个本地 .pt 文件")
    parser.add_argument("--label", default="本地矩阵", help="页面上的 A3 / 950DT 标识")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须大于 0")
    torch.set_num_threads(args.threads)
    matrices = Matrices(args.root, args.file, args.label)
    with HTTPServer((args.host, args.port), handler_for(matrices)) as server:
        print(
            f"{args.label}: {len(matrices.entries)} 个矩阵；http://{args.host}:{server.server_port}",
            flush=True,
        )
        print("仅 CPU 读取，不修改采集文件；Ctrl+C 退出。", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
