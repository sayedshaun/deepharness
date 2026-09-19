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
from deepharness import Agent, Message
from deepharness.providers import Image, Text

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
from deepharness import TextDelta
from deepharness.providers import ThinkingDelta

async for event in agent.astream_events("Prove it"):
    match event:
        case ThinkingDelta(text):
            print(f"\033[2m{text}\033[0m", end="")  # dim: it is working, not answering
        case TextDelta(text):
            print(text, end="")
```

`response.thinking` has the whole of it after the fact, and `state.messages` keeps it as a
`Thinking` block on the assistant turn.

Chat Completions has no field for reasoning, so a server that reasons invents one: llama.cpp
and DeepSeek send `reasoning_content`, OpenRouter sends `reasoning`. Both spellings are read,
in the response and in the stream, so an OpenAI-compatible reasoning model surfaces its
thinking here without any configuration — `Gemini` uses `thought` parts and `Anthropic` its own
thinking blocks, and all three arrive as the same `ThinkingDelta`. That last part matters on Anthropic: it requires the
thinking block back, with its signature, on the request that follows a tool call — so a run
that dropped it would lose the model's chain exactly where a long task depends on it. Thinking
is replayed only where it is required and accepted; OpenAI and Gemini get text alone.
`astream()` never yields reasoning, so a caller that only prints text is unaffected.

## Prompt caching and concurrency

Two knobs for what a harness does differently from a chat: it sends the same long prefix every
turn, and it fans out.

```python
llm = Anthropic("claude-3-5-sonnet-20241022", cache_prompt=True, max_concurrency=8)
```

`cache_prompt=True` (Anthropic) puts a cache breakpoint on the system prompt and the last tool
definition, which covers the part of a request that is identical on every turn of every run.
Most other vendors cache prefixes automatically and charge less for them.

What it saved shows up in usage: `usage.cached_tokens` is the part of `prompt_tokens` served
from the vendor's cache, and `usage.cache_write_tokens` is what it charged to put a prefix
there. Both sit *inside* `prompt_tokens` rather than being deducted from it — a cached token
still occupies the context window — so `Budget(tokens=…)` counts the full figure and these two
are there to tell a cheap turn from an expensive one.

```python
state = await agent.arun("…")
print(state.usage.prompt_tokens, state.usage.cached_tokens)  # 383 377
```

This is not the same as [`Caching`](#caching), which skips the request entirely: prompt caching
makes a *new* request cheaper by reusing its prefix, and helps a conversation that keeps
growing — exactly where a response cache cannot.

`max_concurrency=` caps requests in flight for that provider. A graph wave or a `DeepResearch`
fan-out otherwise opens as many connections as it has branches, which is the usual way a run
rate-limits itself; the cap is held across retries and for a stream's whole body, because
in-flight requests are what a rate limiter counts.

!!! note "Anthropic's wire format"
    Anthropic differs more from Gemini/OpenAI than they differ from each other: the system
    prompt is a separate top-level field, and there's no `role: "tool"` — tool calls/results
    become content blocks instead. `Message` carries the vendor's call id under the hood, so
    this round-trips correctly across turns for all three providers.

## Wrapping a provider

Caching, rate limiting, retrying and falling back are the same shape: do something around a
model call, then delegate. `LLM` is already a narrow interface, so each of these is another
implementation of it — which means one wrapper covers `generate()`, `agenerate()` and both
streaming paths, and they compose at the call site where the order is visible.

```python
from deepharness import Anthropic, OpenAI
from deepharness.providers import Caching, Fallback, RateLimited, Retrying

llm = Caching(
    RateLimited(
        Retrying(Fallback(OpenAI("gpt-4o-mini"), Anthropic("claude-sonnet-4-5"))),
        rps=2,
    )
)
```

Read it outside-in: the cache is asked first, then the limiter, then the retry, and the
fallback is what actually talks to a vendor. `llm.inner` reaches the provider underneath, and
`aclose()`/`close()` pass all the way down.

### `Fallback`

```python
from deepharness.errors import ProviderError
from deepharness.providers import Fallback

llm = Fallback(primary, backup, on=(ProviderError,))  # on= is the default
```

Tries each provider in order. `on` is `ProviderError` rather than `Exception` on purpose:
catching everything would turn a `TypeError` in your own code into "the primary model is
flaky" and quietly send your traffic elsewhere. When every provider fails you get one
`ProviderError` naming how many were tried, with the last failure as its `__cause__`.

Streaming has one rule worth knowing: a stream that fails **before its first event** falls
back, and one that fails partway through raises instead. Those deltas are already with the
caller, and starting over would repeat them.

### `Caching`

```python
Caching(llm, maxsize=256, ttl=None)
```

An in-process LRU keyed on the messages *and* the tool schemas. Worth putting in front of a
deterministic setup — temperature 0, or a classify/extract step. In front of a sampling model
it hands every caller the first answer it happened to get, which is not what sampling is for,
so it is never on by default.

A streamed hit is replayed in block order and ends with the same `Completed`, so a caller's
loop looks identical either way. Responses are copied on the way out, so editing what you got
cannot corrupt the entry behind it. `hits`, `misses` and `clear()` are there for when you want
to know whether it is earning its place.

Two things it deliberately does not do: identical calls made *concurrently* all miss and all
reach the vendor (no single-flight), and nothing is shared between processes.

### `RateLimited`

```python
RateLimited(llm, rps=2, burst=4)
```

A token bucket refilling at `rps`, up to `burst`. A caller reserves its slot when it asks and
then waits out its own turn, so ten simultaneous requests leave in order at the configured rate
rather than retrying against each other.

State is guarded by a `threading.Lock`, not an `asyncio` one — a contended asyncio primitive
binds to the loop that awaited it and raises in any other, and a provider built once at import
time routinely outlives a script's first `asyncio.run()`.

### `Retrying`

```python
Retrying(llm, attempts=2, backoff=0.5)
```

For the other kind of failure: the request succeeded and the model said nothing at all — no
text, no tool call. That turn is unusable, and an agent loop would otherwise spend a step on
it. Transport failures (429, 5xx, a dropped connection) are already retried with backoff by
the HTTP client underneath, so this does not repeat that.

A streamed turn is retried only while nothing has been emitted, which an empty turn satisfies
by definition — so streaming stays as responsive as it was.

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
