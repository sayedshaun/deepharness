# Providers

Every provider implements the same interface, so swapping vendors is a one-line change.

| Method                                          | Returns              |
| ------------------------------------------------ | --------------------- |
| `await agenerate(messages, tools=None)`          | `CompletionResponse`  |
| `generate(messages, tools=None)`                 | `CompletionResponse`  |
| `async for chunk in astream(messages, tools=None)` | text deltas         |
| `for chunk in stream(messages, tools=None)`      | text deltas           |

```python
from deepharness import Anthropic, Gemini, OpenAI

llm = OpenAI(model="gpt-4o-mini", api_key="sk-...")
llm = Gemini(model="gemini-2.0-flash", api_key="...")  # same interface
llm = Anthropic(
    model="claude-3-5-sonnet-20241022", api_key="sk-ant-..."
)  # same interface

response = await llm.agenerate([{"role": "user", "content": "Hi"}])
print(response.content, response.tool_calls)

async for chunk in llm.astream([{"role": "user", "content": "Write a haiku"}]):
    print(chunk, end="", flush=True)
```

Responses normalize to a vendor-neutral `CompletionResponse(content: str, tool_calls:
list[ToolCall], usage: TokenUsage | None, blocks: list[Block])`.

!!! note
    `astream()`/`stream()` yield **text only**. Tool calls are resolved through `generate` /
    `agenerate`, and `astream_events()` carries the rest — reasoning included.

## Content blocks

A message's content is a string in the common case, and a list of blocks when text is not
enough: an image to look at, a file to read, or the model's own reasoning coming back.

```python
from deepharness import Agent, Image, Message, Text

state = await agent.arun(
    [
        Message.human(
            [Text("What changed in this screenshot?"), Image.from_path("ui.png")]
        )
    ]
)
```

`Message.human()` takes a single block too, so `Message.human(Image.from_path("ui.png"))`
works. `Image.from_path()` encodes the file and infers its media type from the suffix;
`Image.from_url()` passes a URL through, and `Document.from_path("report.pdf")` sends a file.
Each provider renders these into its own wire shape — an `image_url` part for OpenAI, a
base64 `source` for Anthropic, `inline_data` for Gemini. Gemini takes no image URLs, so one
raises `ConfigurationError` rather than being dropped silently.

Content that is only text is still sent as a plain string, so an ordinary conversation's
payload — and a saved session — is byte-for-byte what it was before blocks existed.

## Reasoning

`reasoning_effort=` asks a model to think first (`"low"`, `"medium"`, `"high"`; Anthropic and
Gemini take the matching token budget). What comes back is kept apart from the answer:

```python
async for event in agent.astream_events("Prove it"):
    match event:
        case ThinkingDelta(text):
            print(f"\033[2m{text}\033[0m", end="")  # dim: it is working, not answering
        case TextDelta(text):
            print(text, end="")
```

`response.thinking` has the whole of it after the fact, and `state.messages` keeps it as a
`Thinking` block on the assistant turn. That last part matters on Anthropic: it requires the
thinking block back, with its signature, on the request that follows a tool call — so a run
that dropped it would lose the model's chain exactly where a long task depends on it. Thinking
is replayed only where it is required and accepted; OpenAI and Gemini get text alone.
`astream()` never yields reasoning, so a caller that only prints text is unaffected.

## Caching and concurrency

Two knobs for what a harness does differently from a chat: it sends the same long prefix every
turn, and it fans out.

```python
llm = Anthropic("claude-3-5-sonnet-20241022", cache_prompt=True, max_concurrency=8)
```

`cache_prompt=True` (Anthropic) puts a cache breakpoint on the system prompt and the last tool
definition, which covers the part of a request that is identical on every turn of every run.

`max_concurrency=` caps requests in flight for that provider. A graph wave or a `DeepResearch`
fan-out otherwise opens as many connections as it has branches, which is the usual way a run
rate-limits itself; the cap is held across retries and for a stream's whole body, because
in-flight requests are what a rate limiter counts.

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
from deepharness import Groq

model = Groq(
    "llama-3.3-70b-versatile", temperature=0
)  # reads GROQ_API_KEY automatically
```

For a gateway that isn't listed, construct `OpenAI` directly with an explicit `base_url`:

```python
from deepharness import OpenAI

model = OpenAI("llama-3.3-70b", base_url="https://llm.internal/v1", api_key=key)
```

See the [API reference](../reference/api.md#providers) for the full list of gateway classes
and constructor parameters.
