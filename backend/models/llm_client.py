"""
Unified LLM client — supports OpenAI, Ollama, Together.ai, and Groq.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings


class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Provider-agnostic LLM client."""

    def __init__(self):
        self.s = get_settings()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def complete(self, messages: list[Message],
                       temperature: float = 0.1,
                       max_tokens: int = 2048) -> str:
        """Send messages and return the assistant reply text."""
        provider = self.s.llm_provider.lower()

        if provider == "openai":
            return await self._openai(messages, temperature, max_tokens)
        elif provider == "ollama":
            return await self._ollama(messages, temperature, max_tokens)
        elif provider == "together":
            return await self._together(messages, temperature, max_tokens)
        elif provider == "groq":
            return await self._groq(messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def complete_sync(self, messages: list[Message], **kwargs) -> str:
        """Synchronous wrapper around async complete()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.complete(messages, **kwargs))
                    return future.result()
            return loop.run_until_complete(self.complete(messages, **kwargs))
        except RuntimeError:
            return asyncio.run(self.complete(messages, **kwargs))

    # ── Providers ─────────────────────────────────────────────────────────────

    async def _openai(self, messages: list[Message],
                      temperature: float, max_tokens: int) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.s.openai_api_key)
        resp = await client.chat.completions.create(
            model=self.s.openai_model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    async def _ollama(self, messages: list[Message],
                      temperature: float, max_tokens: int) -> str:
        payload = {
            "model":    self.s.ollama_model,
            "messages": [m.to_dict() for m in messages],
            "stream":   False,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.s.ollama_base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

    async def _together(self, messages: list[Message],
                        temperature: float, max_tokens: int) -> str:
        payload = {
            "model":       self.s.together_model,
            "messages":    [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.s.together_api_key}"}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                json=payload, headers=headers,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _groq(self, messages: list[Message],
                    temperature: float, max_tokens: int) -> str:
        payload = {
            "model":       self.s.groq_model,
            "messages":    [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.s.groq_api_key}"}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload, headers=headers,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
