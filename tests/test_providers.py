from unittest.mock import AsyncMock, MagicMock

from subagents.providers.anthropic import Anthropic
from subagents.providers.base import LLM, CompletionResponse, ToolCall
from subagents.providers.gemini import Gemini
from subagents.providers.openai import OpenAI


def make_client(json_body):
    response = AsyncMock()
    response.raise_for_status = lambda: None
    response.json = lambda: json_body
    client = AsyncMock()
    client.post = AsyncMock(return_value=response)
    return client


def make_sync_client(json_body):
    response = MagicMock()
    response.raise_for_status = lambda: None
    response.json = lambda: json_body
    client = MagicMock()
    client.post = MagicMock(return_value=response)
    return client


def make_async_stream_client(lines):
    async def aiter_lines():
        for line in lines:
            yield line

    response = MagicMock()
    response.raise_for_status = lambda: None
    response.aiter_lines = aiter_lines

    class _StreamContext:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *exc_info):
            return False

    client = MagicMock()
    client.stream = MagicMock(return_value=_StreamContext())
    return client


def make_sync_stream_client(lines):
    response = MagicMock()
    response.raise_for_status = lambda: None
    response.iter_lines = lambda: iter(lines)

    class _StreamContext:
        def __enter__(self):
            return response

        def __exit__(self, *exc_info):
            return False

    client = MagicMock()
    client.stream = MagicMock(return_value=_StreamContext())
    return client


async def test_openai_complete_returns_content():
    client = make_client(
        {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    )
    provider = OpenAI(model="gpt-test", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.tool_calls == []
    client.post.assert_awaited_once_with(
        "/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    )


async def test_openai_parses_usage():
    client = make_client(
        {
            "choices": [{"message": {"content": "hello", "tool_calls": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    provider = OpenAI(model="gpt-test", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


async def test_openai_usage_is_none_when_absent():
    client = make_client(
        {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    )
    provider = OpenAI(model="gpt-test", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.usage is None


async def test_openai_complete_passes_tools_and_parses_tool_calls():
    client = make_client(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "cats"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    provider = OpenAI(model="gpt-test", client=client)
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    result = await provider.agenerate([{"role": "user", "content": "hi"}], tools=tools)

    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "cats"}
    sent_kwargs = client.post.await_args.kwargs
    assert sent_kwargs["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


async def test_openai_reconstructs_assistant_tool_call_turn_and_links_result():
    client = make_client(
        {"choices": [{"message": {"content": "done", "tool_calls": None}}]}
    )
    provider = OpenAI(model="gpt-test", client=client)

    await provider.agenerate(
        [
            {"role": "user", "content": "search cats"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "search", "arguments": {"query": "cats"}}
                ],
            },
            {
                "role": "tool",
                "name": "search",
                "content": "found 3 cats",
                "tool_call_id": "call_1",
            },
        ]
    )

    sent_messages = client.post.await_args.kwargs["json"]["messages"]
    assert sent_messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "cats"}'},
            }
        ],
    }
    assert sent_messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "found 3 cats",
    }


async def test_gemini_complete_returns_content():
    client = make_client({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})
    provider = Gemini(model="gemini-test", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.tool_calls == []
    client.post.assert_awaited_once_with(
        "/models/gemini-test:generateContent",
        params={"key": None},
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )


async def test_gemini_parses_usage():
    client = make_client(
        {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 4,
                "totalTokenCount": 12,
            },
        }
    )
    provider = Gemini(model="gemini-test", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.usage.prompt_tokens == 8
    assert result.usage.completion_tokens == 4
    assert result.usage.total_tokens == 12


def test_openai_generate_returns_content_sync():
    sync_client = make_sync_client(
        {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    )
    provider = OpenAI(model="gpt-test", client=AsyncMock(), sync_client=sync_client)

    result = provider.generate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    sync_client.post.assert_called_once_with(
        "/chat/completions",
        json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    )


def test_gemini_generate_returns_content_sync():
    sync_client = make_sync_client(
        {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    )
    provider = Gemini(model="gemini-test", client=AsyncMock(), sync_client=sync_client)

    result = provider.generate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    sync_client.post.assert_called_once_with(
        "/models/gemini-test:generateContent",
        params={"key": None},
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )


async def test_openai_astream_yields_content_deltas():
    client = make_async_stream_client(
        [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
    )
    provider = OpenAI(model="gpt-test", client=client)

    chunks = [
        chunk async for chunk in provider.astream([{"role": "user", "content": "hi"}])
    ]

    assert chunks == ["Hel", "lo"]
    sent_args = client.stream.call_args.args
    sent_kwargs = client.stream.call_args.kwargs
    assert sent_args == ("POST", "/chat/completions")
    assert sent_kwargs["json"]["stream"] is True


def test_openai_stream_yields_content_deltas_sync():
    sync_client = make_sync_stream_client(
        [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
    )
    provider = OpenAI(model="gpt-test", client=AsyncMock(), sync_client=sync_client)

    chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

    assert chunks == ["Hel", "lo"]


async def test_gemini_astream_yields_content_deltas():
    client = make_async_stream_client(
        [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}',
        ]
    )
    provider = Gemini(model="gemini-test", client=client)

    chunks = [
        chunk async for chunk in provider.astream([{"role": "user", "content": "hi"}])
    ]

    assert chunks == ["Hel", "lo"]
    sent_args = client.stream.call_args.args
    sent_kwargs = client.stream.call_args.kwargs
    assert sent_args == ("POST", "/models/gemini-test:streamGenerateContent")
    assert sent_kwargs["params"] == {"key": None, "alt": "sse"}


def test_gemini_stream_yields_content_deltas_sync():
    sync_client = make_sync_stream_client(
        [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}',
        ]
    )
    provider = Gemini(model="gemini-test", client=AsyncMock(), sync_client=sync_client)

    chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

    assert chunks == ["Hel", "lo"]


async def test_anthropic_complete_returns_content():
    client = make_client({"content": [{"type": "text", "text": "hello"}]})
    provider = Anthropic(model="claude-test", api_key="x", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    assert result.tool_calls == []
    client.post.assert_awaited_once_with(
        "/messages",
        json={
            "model": "claude-test",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


async def test_anthropic_parses_usage():
    client = make_client(
        {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }
    )
    provider = Anthropic(model="claude-test", api_key="x", client=client)

    result = await provider.agenerate([{"role": "user", "content": "hi"}])

    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 6
    assert result.usage.total_tokens == 18


async def test_anthropic_moves_system_message_to_top_level_field():
    client = make_client({"content": [{"type": "text", "text": "hello"}]})
    provider = Anthropic(model="claude-test", api_key="x", client=client)

    await provider.agenerate(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]
    )

    sent_kwargs = client.post.await_args.kwargs
    assert sent_kwargs["json"]["system"] == "be helpful"
    assert sent_kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]


async def test_anthropic_reconstructs_assistant_tool_use_turn_and_links_result():
    client = make_client({"content": [{"type": "text", "text": "hello"}]})
    provider = Anthropic(model="claude-test", api_key="x", client=client)

    await provider.agenerate(
        [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "toolu_1", "name": "search", "arguments": {"query": "cats"}}
                ],
            },
            {
                "role": "tool",
                "name": "search",
                "content": "found 3 cats",
                "tool_call_id": "toolu_1",
            },
        ]
    )

    sent_messages = client.post.await_args.kwargs["json"]["messages"]
    assert sent_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search",
                "input": {"query": "cats"},
            }
        ],
    }
    assert sent_messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "found 3 cats"}
        ],
    }


async def test_anthropic_complete_passes_tools_and_parses_tool_calls():
    client = make_client(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search",
                    "input": {"query": "cats"},
                },
            ]
        }
    )
    provider = Anthropic(model="claude-test", api_key="x", client=client)
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    result = await provider.agenerate([{"role": "user", "content": "hi"}], tools=tools)

    assert result.tool_calls[0].id == "toolu_1"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "cats"}
    sent_kwargs = client.post.await_args.kwargs
    assert sent_kwargs["json"]["tools"] == [
        {
            "name": "search",
            "description": "Search the web",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_anthropic_generate_returns_content_sync():
    sync_client = make_sync_client({"content": [{"type": "text", "text": "hello"}]})
    provider = Anthropic(
        model="claude-test", api_key="x", client=AsyncMock(), sync_client=sync_client
    )

    result = provider.generate([{"role": "user", "content": "hi"}])

    assert result.content == "hello"
    sync_client.post.assert_called_once_with(
        "/messages",
        json={
            "model": "claude-test",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


async def test_anthropic_astream_yields_content_deltas():
    client = make_async_stream_client(
        [
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}',
            'data: {"type":"message_stop"}',
        ]
    )
    provider = Anthropic(model="claude-test", api_key="x", client=client)

    chunks = [
        chunk async for chunk in provider.astream([{"role": "user", "content": "hi"}])
    ]

    assert chunks == ["Hel", "lo"]
    sent_args = client.stream.call_args.args
    sent_kwargs = client.stream.call_args.kwargs
    assert sent_args == ("POST", "/messages")
    assert sent_kwargs["json"]["stream"] is True


def test_anthropic_stream_yields_content_deltas_sync():
    sync_client = make_sync_stream_client(
        [
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}',
            'data: {"type":"message_stop"}',
        ]
    )
    provider = Anthropic(
        model="claude-test", api_key="x", client=AsyncMock(), sync_client=sync_client
    )

    chunks = list(provider.stream([{"role": "user", "content": "hi"}]))

    assert chunks == ["Hel", "lo"]


async def test_gemini_complete_passes_tools_and_parses_tool_calls():
    client = make_client(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "cats"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    provider = Gemini(model="gemini-test", client=client)
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    result = await provider.agenerate(
        [{"role": "assistant", "content": "hi"}], tools=tools
    )

    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "cats"}
    sent_kwargs = client.post.await_args.kwargs
    assert sent_kwargs["json"]["contents"] == [
        {"role": "model", "parts": [{"text": "hi"}]}
    ]
    assert sent_kwargs["json"]["tools"] is not None


async def test_openai_sends_reasoning_effort():
    client = make_client(
        {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
    )
    provider = OpenAI(model="gpt-test", client=client, reasoning_effort="high")

    await provider.agenerate([{"role": "user", "content": "hi"}])

    assert client.post.await_args.kwargs["json"]["reasoning_effort"] == "high"


async def test_anthropic_sends_thinking_budget_and_raises_max_tokens():
    client = make_client({"content": [{"type": "text", "text": "hello"}]})
    provider = Anthropic(
        model="claude-test", api_key="x", client=client, reasoning_effort="low"
    )

    await provider.agenerate([{"role": "user", "content": "hi"}])

    sent_json = client.post.await_args.kwargs["json"]
    assert sent_json["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert sent_json["max_tokens"] > 1024


async def test_gemini_sends_thinking_budget():
    client = make_client({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})
    provider = Gemini(model="gemini-test", client=client, reasoning_effort="medium")

    await provider.agenerate([{"role": "user", "content": "hi"}])

    sent_json = client.post.await_args.kwargs["json"]
    assert sent_json["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096


async def test_gemini_reconstructs_assistant_function_call_turn_and_result():
    client = make_client({"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})
    provider = Gemini(model="gemini-test", client=client)

    await provider.agenerate(
        [
            {"role": "user", "content": "search for cats"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": None, "name": "search", "arguments": {"query": "cats"}}
                ],
            },
            {"role": "tool", "name": "search", "content": "found 3 cats"},
        ]
    )

    sent_contents = client.post.await_args.kwargs["json"]["contents"]
    assert sent_contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "search", "args": {"query": "cats"}}}],
    }
    assert sent_contents[2] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "search",
                    "response": {"result": "found 3 cats"},
                }
            }
        ],
    }


async def test_a_text_only_provider_still_streams_in_one_piece():
    """No stub needed: the default turns one completion into one delta."""

    class TextOnly(LLM):
        async def agenerate(self, messages, *, tools=None):
            return CompletionResponse(content="hi")

        def generate(self, messages, *, tools=None):
            return CompletionResponse(content="hi")

    provider = TextOnly()

    assert [chunk async for chunk in provider.astream([])] == ["hi"]
    assert list(provider.stream([])) == ["hi"]


async def test_a_text_only_provider_still_reports_tool_calls_when_streaming():
    class ToolOnly(LLM):
        async def agenerate(self, messages, *, tools=None):
            return CompletionResponse(
                content="", tool_calls=[ToolCall(id="1", name="add", arguments={})]
            )

        def generate(self, messages, *, tools=None):
            raise NotImplementedError

    events = [event async for event in ToolOnly().astream_events([])]

    assert len(events) == 1
    assert events[0].response.tool_calls[0].name == "add"
