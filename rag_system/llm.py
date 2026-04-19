from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaLLMProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided context. "
                        "Cite sources with bracketed numbers such as [1]. "
                        "If the context is insufficient, say so plainly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the Ollama LLM server. "
                "Start Ollama locally or configure LLM_BACKEND with another provider."
            ) from exc

        message = data.get("message", {})
        content = message.get("content", "")
        return str(content).strip()


class OpenAICompatibleLLMProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided context. "
                        "Cite sources with bracketed numbers such as [1]. "
                        "If the context is insufficient, say so plainly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the OpenAI-compatible LLM server. "
                "Check LLM_BASE_URL, network access, and LLM_API_KEY."
            ) from exc

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()


def get_llm_provider(
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
) -> LLMProvider:
    normalized = backend.strip().lower()
    if normalized == "ollama":
        return OllamaLLMProvider(model=model, base_url=base_url, temperature=temperature)
    if normalized in {"vllm", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleLLMProvider(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
    raise ValueError(
        f"Unsupported LLM backend '{backend}'. Add a provider in rag_system/llm.py."
    )
