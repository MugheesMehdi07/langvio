"""
Vision Language Model (VLM) processor implementation using Qwen-VL.

This processor replaces the previous cloud LLM providers (OpenAI, Google Gemini).
It exposes Qwen-VL to the rest of langvio through a small LangChain-compatible
chat model wrapper, so the existing prompt chains in ``BaseLLMProcessor`` work
unchanged.

By default the model is reached over an OpenAI-compatible API with an API key
(see ``langvio.llm.qwen_api``); with ``LANGVIO_BACKEND=local`` the same
processor runs Qwen2-VL / Qwen2.5-VL weights in-process via HuggingFace
transformers instead.

The VLM is used as the language layer of the pipeline: it parses natural language
queries into structured detection parameters and turns the detection summary
produced by the vision layer into a conversational explanation. The vision layer
(``QwenVisionProcessor``) uses the same model for grounding - and, on the local
backend, literally the same shared weights.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langvio.llm.base import BaseLLMProcessor
from langvio.llm.qwen_api import QwenAPIClient, resolve_backend
from langvio.llm.qwen_loader import DEFAULT_VLM_MODEL, GENERATION_LOCK, load_qwen

# Map LangChain message types to the roles Qwen's chat template expects.
_ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}


class QwenChatModel(BaseChatModel):
    """Minimal LangChain chat model wrapping a local Qwen(2/2.5)-VL model.

    Only text is passed to the model - the visual analysis is already done by
    the object/vision model, so the VLM operates purely on the query text and
    the textual detection summary. This keeps it a drop-in replacement for the
    previous ``ChatOpenAI`` / ``ChatGoogleGenerativeAI`` runnables inside the
    prompt chains built by :class:`BaseLLMProcessor`.
    """

    # pydantic fields (BaseChatModel is a pydantic model). ``protected_namespaces``
    # is cleared so a field can carry the model name without a "model_" warning.
    model_config = {"arbitrary_types_allowed": True, "protected_namespaces": ()}

    hf_model: Any = None
    hf_processor: Any = None
    api_client: Any = None
    hf_model_name: str = DEFAULT_VLM_MODEL
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9

    @property
    def _llm_type(self) -> str:
        return "qwen-vl-api" if self.api_client is not None else "qwen-vl-local"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.hf_model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    def _to_qwen_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages into Qwen chat-template messages."""
        qwen_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = _ROLE_MAP.get(message.type, "user")
            content = message.content
            if not isinstance(content, str):
                # Content blocks - keep only the text parts (no images here).
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            qwen_messages.append(
                {"role": role, "content": [{"type": "text", "text": content}]}
            )
        return qwen_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        qwen_messages = self._to_qwen_messages(messages)

        if self.api_client is not None:
            output_text = self.api_client.chat(
                [
                    {"role": m["role"], "content": m["content"][0]["text"]}
                    for m in qwen_messages
                ],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            ).strip()
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=output_text))]
            )

        import torch

        prompt_text = self.hf_processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.hf_processor(text=[prompt_text], return_tensors="pt")
        inputs = inputs.to(self.hf_model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(
                do_sample=True, temperature=self.temperature, top_p=self.top_p
            )
        else:
            gen_kwargs.update(do_sample=False)

        with GENERATION_LOCK, torch.inference_mode():
            generated_ids = self.hf_model.generate(**inputs, **gen_kwargs)

        # Drop the prompt tokens so we decode only the newly generated answer.
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.hf_processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        message = AIMessage(content=output_text)
        return ChatResult(generations=[ChatGeneration(message=message)])


class VLMProcessor(BaseLLMProcessor):
    """LLM processor backed by Qwen-VL over an API key, or local weights."""

    def __init__(
        self,
        name: str = "qwen",
        model_name: str = DEFAULT_VLM_MODEL,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the VLM processor.

        Args:
            name: Processor name.
            model_name: HuggingFace repo id of the Qwen(2/2.5)-VL model. Used by
                the local backend only; the API backend picks its model from
                ``api_model`` / ``QWEN_API_MODEL``.
            model_kwargs: Generation / loading options. Recognized keys:
                temperature, top_p, max_new_tokens (or max_tokens),
                torch_dtype, device_map.
            **kwargs: Additional processor parameters, including the backend
                selector (``backend``) and API options (``api_model``,
                ``api_base_url``, ``api_key_env``, ``api_extra_body``,
                ``api_timeout``, ``api_max_retries``).
        """
        config = {
            "model_name": model_name,
            "model_kwargs": model_kwargs or {},
            **kwargs,
        }
        super().__init__(name, config)
        self.logger = logging.getLogger(__name__)

    def _initialize_llm(self) -> None:
        """Wrap the Qwen API client or the shared local model as a chat model."""
        if resolve_backend(self.config) == "api":
            model_kwargs = self.config["model_kwargs"]
            client = QwenAPIClient.from_config(self.config)
            self.llm = QwenChatModel(
                api_client=client,
                hf_model_name=client.model,
                max_new_tokens=int(
                    model_kwargs.get(
                        "max_new_tokens", model_kwargs.get("max_tokens", 1024)
                    )
                ),
                temperature=float(model_kwargs.get("temperature", 0.2)),
                top_p=float(model_kwargs.get("top_p", 0.9)),
            )
            self.logger.info(
                f"Initialized Qwen API (language): {client.model} via {client.base_url}"
            )
            return

        try:
            if not self.is_package_installed("transformers"):
                raise ImportError(
                    "The 'transformers' package is required to use the local VLM. "
                    "Please install it with 'pip install langvio[vlm]'"
                )

            model_name = self.config["model_name"]
            model_kwargs = dict(self.config["model_kwargs"])

            # Split generation options from model-loading options.
            temperature = float(model_kwargs.pop("temperature", 0.2))
            top_p = float(model_kwargs.pop("top_p", 0.9))
            max_new_tokens = int(
                model_kwargs.pop("max_new_tokens", model_kwargs.pop("max_tokens", 1024))
            )

            # Remaining kwargs (e.g. torch_dtype) go to the shared loader, which
            # returns the same instance the vision processor uses.
            hf_model, hf_processor = load_qwen(
                model_name, load_kwargs=model_kwargs or None
            )

            self.llm: Optional[Any] = QwenChatModel(
                hf_model=hf_model,
                hf_processor=hf_processor,
                hf_model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            self.logger.info(f"Initialized local VLM (language): {model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing local VLM: {e}")
            raise
