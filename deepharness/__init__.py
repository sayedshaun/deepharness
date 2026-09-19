"""A harness for LLM agents: workspace tools, permissions, and typed workflows.

What a first program needs is here; everything else lives in the layer it
belongs to, so a name's import path says which part of the library owns it:

    deepharness.agent      the think/act loop's own types - context policy,
                           progress events, sessions
    deepharness.graph      node specs and the reducers for merging branches
    deepharness.providers  wire types, content blocks, and the wrappers that
                           cache, retry, rate limit or fall back
    deepharness.tools      writing a tool, the workspace tools, permissions,
                           MCP servers, web search
    deepharness.prebuilt   ready-made workflows such as DeepResearch
    deepharness.errors     every exception but the base class

Keeping the root small is deliberate: everything exported here is a promise
about stability, and a flat namespace of everything stops being discoverable
somewhere well before it is complete.
"""

from deepharness.agent import (
    Agent,
    AgentState,
    Budget,
    Ctx,
    Finished,
    Message,
    Toolbox,
    tool,
)
from deepharness.errors import DeepHarnessError
from deepharness.graph import Executor, Graph
from deepharness.providers import (
    VLLM,
    XAI,
    Anthropic,
    Cerebras,
    DeepSeek,
    Fireworks,
    Gemini,
    Groq,
    LlamaCpp,
    LMStudio,
    Mistral,
    Ollama,
    OpenAI,
    OpenRouter,
    TextDelta,
    Together,
)

__all__ = [
    "VLLM",
    "XAI",
    "Agent",
    "AgentState",
    "Anthropic",
    "Budget",
    "Cerebras",
    "Ctx",
    "DeepHarnessError",
    "DeepSeek",
    "Executor",
    "Finished",
    "Fireworks",
    "Gemini",
    "Graph",
    "Groq",
    "LMStudio",
    "LlamaCpp",
    "Message",
    "Mistral",
    "Ollama",
    "OpenAI",
    "OpenRouter",
    "TextDelta",
    "Together",
    "Toolbox",
    "tool",
]
