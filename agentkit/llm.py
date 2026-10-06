"""One interface over every OpenAI-compatible provider, with a fallback chain.

Switching provider = changing an env var. Free tiers rate-limit and flake, so a chain like
    LLM_PROVIDERS=gemini:gemini-2.5-flash,openrouter:openrouter/free
tries the next provider when one is rate-limited, down, or rejects the request.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import openai


class LLMError(Exception):
    """The provider failed (rate limit, outage, unsupported request...)."""


class ConfigError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> Reply: ...


# provider -> (base_url, api-key env var, default model). Model names drift: override with
# "provider:model" in LLM_PROVIDERS and check the provider's model list when in doubt.
PRESETS: dict[str, tuple[str, str, str]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5-mini"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter/free"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", "qwen3:8b"),
}


class OpenAICompatLLM:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        model: str,
        label: str | None = None,
        temperature: float | None = None,
        timeout: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.label = label or model
        # Some newer models reject a non-default temperature, so it is only sent when set.
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> Reply:
        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise LLMError(f"{self.label}: {type(exc).__name__}: {exc}") from exc
        if not response.choices:
            raise LLMError(f"{self.label}: empty response (no choices)")
        message = response.choices[0].message
        calls = [
            ToolCall(
                id=tc.id or f"call_{i}",
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
            )
            for i, tc in enumerate(message.tool_calls or [])
            if getattr(tc, "function", None) is not None
        ]
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
            }
        return Reply(content=message.content, tool_calls=calls, usage=usage, model=self.label)


class FallbackLLM:
    """Try providers in order; a provider that just failed is skipped for `cooldown` seconds."""

    def __init__(self, llms: list[LLM], cooldown: float = 30.0) -> None:
        if not llms:
            raise ConfigError("FallbackLLM needs at least one LLM")
        self.llms = llms
        self.cooldown = cooldown
        self._cool_until = [0.0] * len(llms)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> Reply:
        now = time.monotonic()
        order = [i for i in range(len(self.llms)) if self._cool_until[i] <= now]
        order += [i for i in range(len(self.llms)) if i not in order]  # everything cooling: try anyway
        errors: list[str] = []
        for i in order:
            try:
                return self.llms[i].chat(messages, tools, tool_choice)
            except LLMError as exc:
                self._cool_until[i] = time.monotonic() + self.cooldown
                errors.append(str(exc))
        raise LLMError("all providers failed: " + " | ".join(errors))


def from_env(env: Mapping[str, str] | None = None) -> LLM:
    """Build the LLM from environment variables.

    LLM_PROVIDERS="gemini,openrouter:qwen/qwen3-coder:free"   (comma-separated chain)
      each item is provider or provider:model (split on the first colon only).
    LLM_PROVIDER=openai is accepted for a single provider (default: openai).
    Custom endpoint: provider "custom" with LLM_BASE_URL, LLM_API_KEY and a model.
    LLM_TEMPERATURE optionally sets the temperature.
    """
    env = os.environ if env is None else env
    spec = env.get("LLM_PROVIDERS") or env.get("LLM_PROVIDER") or "openai"
    temperature = float(env["LLM_TEMPERATURE"]) if env.get("LLM_TEMPERATURE") else None
    llms: list[LLM] = []
    skipped: list[str] = []
    for item in [s.strip() for s in spec.split(",") if s.strip()]:
        provider, _, model = item.partition(":")
        provider = provider.lower()
        if provider == "custom":
            base_url, key, default_model = env.get("LLM_BASE_URL"), env.get("LLM_API_KEY"), env.get("LLM_MODEL")
            if not (base_url and key and (model or default_model)):
                skipped.append("custom (needs LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)")
                continue
            llms.append(OpenAICompatLLM(base_url=base_url, api_key=key, model=model or default_model or "", label=f"custom:{model or default_model}", temperature=temperature))
            continue
        if provider not in PRESETS:
            raise ConfigError(f"Unknown provider '{provider}'. Known: {', '.join(PRESETS)}, custom.")
        base_url, key_env, default_model = PRESETS[provider]
        key = env.get(key_env) or ("ollama" if provider == "ollama" else None)
        if not key:
            skipped.append(f"{provider} (set {key_env})")
            continue
        chosen = model or env.get("LLM_MODEL") or default_model
        llms.append(OpenAICompatLLM(base_url=base_url, api_key=key, model=chosen, label=f"{provider}:{chosen}", temperature=temperature))
    if not llms:
        raise ConfigError(
            "No usable LLM provider. Missing: " + (", ".join(skipped) or spec) + ". See .env.example."
        )
    return llms[0] if len(llms) == 1 else FallbackLLM(llms)
