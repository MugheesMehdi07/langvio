"""
LLM processors module for langvio
"""

from langvio.llm.base import BaseLLMProcessor

# The concrete processor (local Qwen2.5-VL VLM) is imported dynamically by the
# factory to avoid requiring the heavy 'transformers' dependency at import time.
# from langvio.llm.qwen_vlm import VLMProcessor

__all__ = ["BaseLLMProcessor"]
