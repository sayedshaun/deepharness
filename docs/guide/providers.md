# Providers

Every provider implements the same interface, so swapping vendors is a one-line change.

| Method                                          | Returns              |
| ------------------------------------------------ | --------------------- |
| `await agenerate(messages, tools=None)`          | `CompletionResponse`  |
| `generate(messages, tools=None)`                 | `CompletionResponse`  |
| `async for chunk in astream(messages, tools=None)` | text deltas         |
| `for chunk in stream(messages, tools=None)`      | text deltas           |

```python
from subagents import Anthropic, Gemini, OpenAI

llm = OpenAI(model="gpt-4o-mini", api_key="sk-...")
llm = Gemini(model="gemini-2.0-flash", api_key="...")           # same interface
llm = Anthropic(model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")  # same interface

response = await llm.agenerate([{"role": "user", "content": "Hi"}])
print(response.content, response.tool_calls)

async for chunk in llm.astream([{"role": "user", "content": "Write a haiku"}]):
    print(chunk, end="", flush=True)
```

Responses normalize to a vendor-neutral `CompletionResponse(content: str, tool_calls: list[ToolCall], usage: TokenUsage | None)`.

!!! note
    Streaming yields **text deltas only**. Tool calls are resolved through `generate` /
    `agenerate`.

!!! note "Anthropic's wire format"
    Anthropic differs more from Gemini/OpenAI than they differ from each other: the system
    prompt is a separate top-level field, and there's no `role: "tool"` — tool calls/results
    become content blocks instead. `Message` carries the vendor's call id under the hood, so
    this round-trips correctly across turns for all three providers.

## OpenAI-compatible gateways

Groq, Together, DeepSeek, Mistral, xAI, OpenRouter, Fireworks, Cerebras, and local servers
(Ollama, vLLM, LM Studio) all speak the same wire format as OpenAI's Chat Completions API —
just a different base URL and API key. Each one is a ready-to-use class that reads its own
API key from the environment automatically:

```python
from subagents import Groq

model = Groq("llama-3.3-70b-versatile", temperature=0)  # reads GROQ_API_KEY automatically
```

For a gateway that isn't listed, construct `OpenAI` directly with an explicit `base_url`:

```python
from subagents import OpenAI

model = OpenAI("llama-3.3-70b", base_url="https://llm.internal/v1", api_key=key)
```

See the [API reference](../reference/api.md#providers) for the full list of gateway classes
and constructor parameters.
