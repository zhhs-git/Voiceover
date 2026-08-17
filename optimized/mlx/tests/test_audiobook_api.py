"""Weight-free regression tests for the audiobook recovery API."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sa3_gradio.py"
MODULE_NAME = "sa3_gradio_audiobook_api_test"


def _load_module():
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[MODULE_NAME] = module
    specification.loader.exec_module(module)
    return module


sa3_gradio = _load_module()


class AudiobookApiTests(unittest.TestCase):
    def test_parse_request_accepts_supported_recovery_payload(self):
        request = sa3_gradio.parse_audiobook_api_request(
            {
                "model": "sm-music",
                "decoder": "same-s",
                "prompt": "quiet ambient score",
                "negativePrompt": "vocals",
                "seconds": 12,
                "cfg": 3,
                "steps": 8,
                "seed": 123,
            }
        )

        self.assertEqual(request["model"], "sm-music")
        self.assertEqual(request["seconds"], 12.0)
        self.assertEqual(request["steps"], 8)
        self.assertEqual(request["seed"], 123)

    def test_parse_request_rejects_out_of_contract_models_and_steps(self):
        with self.assertRaisesRegex(ValueError, "model must be sm-music or sm-sfx"):
            sa3_gradio.parse_audiobook_api_request(
                {
                    "model": "medium",
                    "decoder": "same-s",
                    "prompt": "not allowed",
                    "seconds": 12,
                }
            )
        with self.assertRaisesRegex(ValueError, "steps must be an integer between 1 and 16"):
            sa3_gradio.parse_audiobook_api_request(
                {
                    "model": "sm-sfx",
                    "decoder": "same-s",
                    "prompt": "door closes",
                    "seconds": 2,
                    "steps": 17,
                }
            )

    def test_loopback_check_does_not_accept_remote_clients(self):
        self.assertTrue(sa3_gradio._is_loopback_client("127.0.0.1"))
        self.assertTrue(sa3_gradio._is_loopback_client("::1"))
        self.assertFalse(sa3_gradio._is_loopback_client("192.168.0.24"))
        self.assertFalse(sa3_gradio._is_loopback_client(None))

    def test_generate_response_writes_owned_readable_wav_without_model_weights(self):
        request = {
            "model": "sm-sfx",
            "decoder": "same-s",
            "prompt": "single bell",
            "negativePrompt": "speech",
            "seconds": 1.0,
            "cfg": 3.0,
            "steps": 8,
            "seed": 321,
        }
        samples = np.zeros((2, 4410), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            with (
                mock.patch.object(sa3_gradio, "AUDIOBOOK_OUTPUT_DIR", output_directory),
                mock.patch.object(
                    sa3_gradio,
                    "run_generation",
                    return_value=(samples, {"totalSeconds": 0.01}),
                ),
            ):
                response = sa3_gradio.generate_audiobook_api_wav(request)

            output_path = Path(response["path"])
            self.assertEqual(output_path.parent, output_directory.resolve())
            self.assertTrue(output_path.is_file())
            self.assertEqual(response["durationSeconds"], 0.1)
            with wave.open(str(output_path), "rb") as generated:
                self.assertEqual(generated.getframerate(), sa3_gradio.SAMPLE_RATE)
                self.assertEqual(generated.getnchannels(), 2)


if __name__ == "__main__":
    unittest.main()
