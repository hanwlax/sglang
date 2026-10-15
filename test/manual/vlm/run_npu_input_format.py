"""Run a supplied NPU VLM regression file with explicit model/image paths.

The model and assertions are unchanged. A response recorder retains output token
IDs for comparisons across implementations and hardware. This runner expects
the nine-case file from Ascend/sglang used in the GMM regression investigation.
"""

import argparse
import hashlib
import json
import runpy
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--gemma-model", required=True)
    parser.add_argument("--kimi-model", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qwen-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import sglang
    import sglang.test.ascend.test_ascend_utils as utils

    utils.GEMMA_3_4B_IT_WEIGHTS_PATH = args.gemma_model
    utils.KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH = args.kimi_model
    utils.QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH = args.qwen_model
    utils.IMAGE_MAN_IRONING_PATH = str(
        args.image_dir / "man_ironing_on_back_of_suv.png"
    )
    utils.IMAGE_SGL_LOGO_PATH = str(args.image_dir / "sgl_logo.png")
    namespace = runpy.run_path(str(args.case), run_name="vlm_input_format_case")
    base = namespace["VLMInputTestBase"]
    original_verify = base.verify_response
    responses = []

    def verify_with_record(self, output):
        responses.append({"test": self.id(), "output": output})
        (args.output_dir / "responses.json").write_text(
            json.dumps(responses, ensure_ascii=False, indent=2)
        )
        return original_verify(self, output)

    base.verify_response = verify_with_record
    classes = ["TestQwenVLUnderstandsImage"]
    if not args.qwen_only:
        classes = [
            "TestGemmaUnderstandsImage",
            "TestKimiVLImageUnderstandsImage",
            *classes,
        ]
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(namespace[name])
        for name in classes
    )
    expected = 3 if args.qwen_only else 9
    assert suite.countTestCases() == expected
    result = unittest.TextTestRunner(verbosity=2, failfast=False).run(suite)
    summary = {
        "successful": result.wasSuccessful() and not result.skipped,
        "tests_run": result.testsRun,
        "expected": expected,
        "failures": [(str(t), message) for t, message in result.failures],
        "errors": [(str(t), message) for t, message in result.errors],
        "skipped": [(str(t), message) for t, message in result.skipped],
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "sglang_package": sglang.__file__,
    }
    (args.output_dir / "result.json").write_text(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["successful"] and result.testsRun == expected else 1)


if __name__ == "__main__":
    main()
