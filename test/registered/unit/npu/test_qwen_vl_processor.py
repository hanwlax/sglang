"""NPU patch extraction must preserve Transformers image/video preprocessing.

Transformers 5.12.1 inherits resize(image=..., size=explicit_height_width),
whereas newer Qwen processors have resize(images/videos=..., factor=...).
The old NPU wrappers assumed the latter and failed before patch extraction.
These CPU tests compare against the installed, unpatched HF processors.
"""

import unittest
from unittest.mock import patch

import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
    npu_wrapper_preprocess,
    npu_wrapper_video_preprocess,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwenVLNpuProcessor(CustomTestCase):
    def _check_preprocess(self, processor, inputs, wrapper, **kwargs):
        expected = processor(**inputs, return_tensors="pt", **kwargs)
        cls = type(processor)
        with patch.object(cls, "_preprocess", wrapper(cls._preprocess)):
            actual = processor(**inputs, return_tensors="pt", **kwargs)
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_images_match_hf_resize_and_patch_order(self):
        # Upsize, downsize, then revisit the first shape to exercise regrouping.
        images = [
            torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
            for h, w in [(17, 25), (193, 257), (17, 25)]
        ]
        processor = Qwen2VLImageProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 32 * 32, "longest_edge": 64 * 96},
        )
        self._check_preprocess(processor, {"images": images}, npu_wrapper_preprocess)

    def test_videos_match_hf_resize_padding_and_patch_order(self):
        # Odd frame counts require repeating the final frame before patchifying.
        videos = [
            torch.randint(0, 256, (t, 3, h, w), dtype=torch.uint8)
            for t, h, w in [(3, 65, 97), (5, 129, 193), (3, 65, 97)]
        ]
        processor = Qwen3VLVideoProcessor(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size={"shortest_edge": 32 * 32, "longest_edge": 4 * 64 * 96},
        )
        self._check_preprocess(
            processor,
            {"videos": videos},
            npu_wrapper_video_preprocess,
            do_sample_frames=False,
        )


if __name__ == "__main__":
    unittest.main()
