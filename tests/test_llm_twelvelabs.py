"""
Unit tests for the TwelveLabs Pegasus LLM processor.

The no-network tests run everywhere. The live test is skipped unless a real
TWELVELABS_API_KEY is present in the environment.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from langvio.llm.twelvelabs import PegasusProcessor


class TestPegasusProcessor(unittest.TestCase):
    """Test cases for PegasusProcessor (no network)."""

    def test_initialization(self):
        processor = PegasusProcessor("test", model_name="pegasus1.5")
        self.assertEqual(processor.name, "test")
        self.assertEqual(processor.config["model_name"], "pegasus1.5")

    def test_initialization_with_model_kwargs(self):
        processor = PegasusProcessor(
            "test", model_name="pegasus1.5", model_kwargs={"max_tokens": 4096}
        )
        self.assertIn("model_kwargs", processor.config)
        self.assertEqual(processor.config["model_kwargs"]["max_tokens"], 4096)

    def test_parse_query_task_types(self):
        """Keyword-based query parsing maps to sensible task types."""
        processor = PegasusProcessor("test")
        # Avoid touching the network during init.
        processor.client = object()
        self.assertEqual(
            processor.parse_query("count the cars")["task_type"], "counting"
        )
        self.assertTrue(processor.parse_query("how many people?")["count_objects"])
        self.assertEqual(
            processor.parse_query("track the runner")["task_type"], "tracking"
        )
        self.assertEqual(
            processor.parse_query("what is the dog doing")["task_type"], "activity"
        )
        # Falls back to identification for generic queries.
        self.assertEqual(
            processor.parse_query("describe the scene")["task_type"], "identification"
        )

    def test_analyze_video_with_url(self):
        """analyze_video wires a URL into a VideoContext_Url and returns data."""
        processor = PegasusProcessor("test", model_kwargs={"max_tokens": 2048})
        fake_client = MagicMock()
        fake_client.analyze.return_value = MagicMock(data="a cat plays")
        processor.client = fake_client

        out = processor.analyze_video(
            "https://example.com/clip.mp4", "What happens here?"
        )

        self.assertEqual(out, "a cat plays")
        kwargs = fake_client.analyze.call_args.kwargs
        self.assertEqual(kwargs["model_name"], "pegasus1.5")
        self.assertEqual(kwargs["prompt"], "What happens here?")
        # max_tokens floor of 512 is respected.
        self.assertGreaterEqual(kwargs["max_tokens"], 512)
        self.assertEqual(kwargs["video"].url, "https://example.com/clip.mp4")

    def test_analyze_video_enforces_min_max_tokens(self):
        processor = PegasusProcessor("test", model_kwargs={"max_tokens": 10})
        fake_client = MagicMock()
        fake_client.analyze.return_value = MagicMock(data="ok")
        processor.client = fake_client

        processor.analyze_video("https://example.com/clip.mp4", "q")

        self.assertEqual(fake_client.analyze.call_args.kwargs["max_tokens"], 512)

    def test_generate_explanation_uses_pegasus_for_video(self):
        processor = PegasusProcessor("test")
        processor.client = object()
        with patch.object(
            processor, "analyze_video", return_value="server-side answer"
        ) as mock_av, patch("langvio.llm.twelvelabs.is_video_file", return_value=True):
            out = processor.generate_explanation(
                "describe", {"frame_detections": {}}, is_video=True, media_path="x.mp4"
            )
        self.assertEqual(out, "server-side answer")
        mock_av.assert_called_once_with("x.mp4", "describe")


@unittest.skipUnless(
    os.getenv("TWELVELABS_API_KEY"), "TWELVELABS_API_KEY not set; skipping live test"
)
class TestPegasusLive(unittest.TestCase):
    """Live smoke test against the TwelveLabs API (requires a real key)."""

    def test_analyze_public_video(self):
        processor = PegasusProcessor("pegasus", model_name="pegasus1.5")
        url = (
            "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
            "Big_Buck_Bunny_360_10s_1MB.mp4"
        )
        out = processor.analyze_video(url, "Describe this video in one sentence.")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out.strip()), 0)


if __name__ == "__main__":
    unittest.main()
