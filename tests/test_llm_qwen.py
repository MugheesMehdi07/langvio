"""
Tests for the local Qwen2.5-VL VLM processor.

These tests exercise the processor's configuration and the LangChain chat-model
wrapper without actually downloading or running the (multi-GB) model weights.
"""

import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from langvio.llm.qwen_vlm import DEFAULT_VLM_MODEL, QwenChatModel, VLMProcessor


class TestVLMProcessor(unittest.TestCase):
    """Test cases for VLMProcessor configuration."""

    def test_default_initialization(self):
        """VLMProcessor stores its model name and kwargs in config."""
        processor = VLMProcessor("test")
        self.assertEqual(processor.config["model_name"], DEFAULT_VLM_MODEL)
        self.assertEqual(processor.config["model_kwargs"], {})

    def test_initialization_with_kwargs(self):
        """Custom model name and generation kwargs are preserved."""
        config = {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_kwargs": {"temperature": 0.5, "max_new_tokens": 512},
        }
        processor = VLMProcessor("test", **config)
        self.assertEqual(processor.config["model_name"], config["model_name"])
        self.assertEqual(processor.config["model_kwargs"]["temperature"], 0.5)


class TestQwenChatModel(unittest.TestCase):
    """Test cases for the LangChain chat-model wrapper (no model loaded)."""

    def test_llm_type(self):
        model = QwenChatModel()
        self.assertEqual(model._llm_type, "qwen-vl-local")

    def test_message_conversion(self):
        """LangChain messages map to Qwen chat-template roles."""
        model = QwenChatModel()
        qwen_messages = model._to_qwen_messages(
            [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="How many cars?"),
            ]
        )
        self.assertEqual(qwen_messages[0]["role"], "system")
        self.assertEqual(qwen_messages[1]["role"], "user")
        self.assertEqual(
            qwen_messages[1]["content"][0]["text"], "How many cars?"
        )


if __name__ == "__main__":
    unittest.main()
