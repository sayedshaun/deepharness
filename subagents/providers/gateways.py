"""OpenAI-compatible endpoints.

Every class here is the same wire format as :class:`~subagents.providers.OpenAI`
with a different base URL and credential. That is the entire difference, which is
why each one fits in four lines::

    from subagents import Groq

    model = Groq("llama-3.3-70b-versatile", temperature=0)

For a gateway that is not listed, construct ``OpenAI`` against it directly --
there is nothing to register::

    OpenAI("llama-3.3-70b", base_url="https://llm.internal/v1", api_key=key)
"""

from __future__ import annotations

from .openai import OpenAI


class Groq(OpenAI):
    provider = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    env_key = "GROQ_API_KEY"
    __slots__ = ()


class Together(OpenAI):
    provider = "together"
    default_base_url = "https://api.together.xyz/v1"
    env_key = "TOGETHER_API_KEY"
    __slots__ = ()


class DeepSeek(OpenAI):
    provider = "deepseek"
    default_base_url = "https://api.deepseek.com/v1"
    env_key = "DEEPSEEK_API_KEY"
    __slots__ = ()


class Mistral(OpenAI):
    provider = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    env_key = "MISTRAL_API_KEY"
    __slots__ = ()


class XAI(OpenAI):
    provider = "xai"
    default_base_url = "https://api.x.ai/v1"
    env_key = "XAI_API_KEY"
    __slots__ = ()


class OpenRouter(OpenAI):
    provider = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    env_key = "OPENROUTER_API_KEY"
    __slots__ = ()


class Fireworks(OpenAI):
    provider = "fireworks"
    default_base_url = "https://api.fireworks.ai/inference/v1"
    env_key = "FIREWORKS_API_KEY"
    __slots__ = ()


class Cerebras(OpenAI):
    provider = "cerebras"
    default_base_url = "https://api.cerebras.ai/v1"
    env_key = "CEREBRAS_API_KEY"
    __slots__ = ()


# Local servers: no credential, so ``env_key`` is empty rather than unset.


class Ollama(OpenAI):
    provider = "ollama"
    default_base_url = "http://localhost:11434/v1"
    env_key = ""
    __slots__ = ()


class VLLM(OpenAI):
    provider = "vllm"
    default_base_url = "http://localhost:8000/v1"
    env_key = ""
    __slots__ = ()


class LMStudio(OpenAI):
    provider = "lmstudio"
    default_base_url = "http://localhost:1234/v1"
    env_key = ""
    __slots__ = ()
