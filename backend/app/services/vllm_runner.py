"""vLLM AsyncLLMEngine wrapper. Lazy-imports vllm so the server can boot without it."""
from __future__ import annotations

import asyncio
import uuid
from typing import AsyncIterator, Optional

from app.schemas import ChatMessage, LoadModelRequest


class VLLMRunner:
    def __init__(self) -> None:
        self._engine = None
        self._tokenizer = None
        self._loaded_path: Optional[str] = None
        self._loaded_quant: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def loaded_path(self) -> Optional[str]:
        return self._loaded_path

    @property
    def loaded_quant(self) -> Optional[str]:
        return self._loaded_quant

    async def load(self, req: LoadModelRequest) -> None:
        async with self._lock:
            if self._engine is not None:
                await self._unload_locked()

            from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore

            args = AsyncEngineArgs(
                model=req.path,
                dtype=req.dtype,
                max_model_len=req.max_model_len,
                gpu_memory_utilization=req.gpu_memory_utilization,
                quantization=req.quantization,
            )
            self._engine = AsyncLLMEngine.from_engine_args(args)
            self._loaded_path = req.path
            self._loaded_quant = req.quantization

    async def unload(self) -> None:
        async with self._lock:
            await self._unload_locked()

    async def _unload_locked(self) -> None:
        if self._engine is None:
            return
        try:
            shutdown = getattr(self._engine, "shutdown", None)
            if shutdown:
                res = shutdown()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass
        self._engine = None
        self._loaded_path = None
        self._loaded_quant = None
        try:
            import gc, torch  # type: ignore
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _format_prompt(self, messages: list[ChatMessage]) -> str:
        # Try the model's chat template via tokenizer; fall back to a simple format.
        try:
            from transformers import AutoTokenizer  # type: ignore

            if self._tokenizer is None and self._loaded_path:
                self._tokenizer = AutoTokenizer.from_pretrained(self._loaded_path)
            if self._tokenizer is not None and getattr(self._tokenizer, "chat_template", None):
                return self._tokenizer.apply_chat_template(
                    [m.model_dump() for m in messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass
        parts = [f"{m.role}: {m.content}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[dict]:
        """Yields events:
        - {"type": "delta", "text": "..."} for each new chunk
        - {"type": "usage", "prompt_tokens": N, "completion_tokens": M} once at end
        """
        if self._engine is None:
            raise RuntimeError("No model loaded")

        from vllm import SamplingParams  # type: ignore

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        prompt = self._format_prompt(messages)
        request_id = str(uuid.uuid4())

        previous = ""
        # Track tokens incrementally so we never rely on the very last
        # RequestOutput having a populated `outputs` list.
        prompt_tokens = 0
        completion_tokens = 0
        async for out in self._engine.generate(prompt, sp, request_id):
            try:
                ptids = getattr(out, "prompt_token_ids", None)
                if ptids:
                    prompt_tokens = len(ptids)
            except Exception:
                pass
            if not out.outputs:
                continue
            try:
                tids = getattr(out.outputs[0], "token_ids", None)
                if tids is not None:
                    completion_tokens = max(completion_tokens, len(tids))
            except Exception:
                pass
            text = out.outputs[0].text
            if len(text) > len(previous):
                yield {"type": "delta", "text": text[len(previous):]}
                previous = text

        # Always emit usage so the client can finalize, even if token counts
        # could not be determined (in which case they'll be 0).
        yield {
            "type": "usage",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


runner = VLLMRunner()
