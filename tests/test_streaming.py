"""Streaming: vendor accumulators, then a whole agent run streamed."""

from subagents.agent import Agent, tool
from subagents.agent.loop import Finished
from subagents.providers.anthropic import AnthropicStream
from subagents.providers.base import (
    LLM,
    Completed,
    CompletionResponse,
    TextDelta,
    TokenUsage,
)
from subagents.providers.gemini import GeminiStream
from subagents.providers.openai import OpenAIStream


def feed_all(reader, payloads):
    return [reader.feed(payload) for payload in payloads]


def test_openai_stream_collects_text():
    reader = OpenAIStream()

    deltas = feed_all(
        reader,
        [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}}]},
        ],
    )

    assert deltas == ["Hel", "lo", None]
    assert reader.response() == CompletionResponse(content="Hello")


def test_openai_stream_reassembles_a_fragmented_tool_call():
    reader = OpenAIStream()

    feed_all(
        reader,
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "add", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"a": 1'}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ', "b": 2}'}}
                            ]
                        }
                    }
                ]
            },
            {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
        ],
    )
    response = reader.response()

    assert response.content == ""
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("call_1", "add", {"a": 1, "b": 2})
    assert response.usage == TokenUsage(5, 7, 12)


def test_openai_stream_keeps_two_tool_calls_apart_by_index():
    reader = OpenAIStream()

    feed_all(
        reader,
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "function": {"name": "one", "arguments": "{}"},
                                },
                                {
                                    "index": 1,
                                    "id": "b",
                                    "function": {"name": "two", "arguments": "{}"},
                                },
                            ]
                        }
                    }
                ]
            }
        ],
    )

    assert [c.name for c in reader.response().tool_calls] == ["one", "two"]


def test_truncated_tool_arguments_do_not_break_the_stream():
    reader = OpenAIStream()

    feed_all(
        reader,
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "add", "arguments": '{"a": '},
                                }
                            ]
                        }
                    }
                ]
            }
        ],
    )

    assert reader.response().tool_calls[0].arguments == {}


def test_anthropic_stream_collects_text_and_usage():
    reader = AnthropicStream()

    deltas = feed_all(
        reader,
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "usage": {"input_tokens": 4, "output_tokens": 6}},
        ],
    )
    response = reader.response()

    assert [d for d in deltas if d] == ["Hi"]
    assert response.content == "Hi"
    assert response.usage == TokenUsage(4, 6, 10)


def test_anthropic_stream_reassembles_tool_use_blocks():
    reader = AnthropicStream()

    feed_all(
        reader,
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tu_1", "name": "add"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"a":'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": " 2}"},
            },
            {"type": "content_block_stop", "index": 0},
        ],
    )
    call = reader.response().tool_calls[0]

    assert (call.id, call.name, call.arguments) == ("tu_1", "add", {"a": 2})


def test_anthropic_text_and_tool_blocks_coexist():
    reader = AnthropicStream()

    feed_all(
        reader,
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "let me check"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tu_1", "name": "add"},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        ],
    )
    response = reader.response()

    assert response.content == "let me check"
    assert [c.name for c in response.tool_calls] == ["add"]


def test_gemini_stream_collects_text_and_whole_function_calls():
    reader = GeminiStream()

    deltas = feed_all(
        reader,
        [
            {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "add", "args": {"a": 1}}}
                            ]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 7,
                },
            },
        ],
    )
    response = reader.response()

    assert [d for d in deltas if d] == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.tool_calls[0].arguments == {"a": 1}
    assert response.usage == TokenUsage(3, 4, 7)


class StreamingProvider(LLM):
    """Replays scripted turns as streams, one turn per call."""

    def __init__(self, turns: list[tuple[list[str], CompletionResponse]]):
        self.turns = turns
        self.calls: list[list[dict]] = []

    async def agenerate(self, messages, *, tools=None):
        raise AssertionError("streaming test should not call agenerate")

    def generate(self, messages, *, tools=None):
        raise AssertionError("streaming test should not call generate")

    async def astream_events(self, messages, *, tools=None):
        self.calls.append([dict(m) for m in messages])
        deltas, response = self.turns[len(self.calls) - 1]
        for delta in deltas:
            yield TextDelta(delta)
        yield Completed(response)

    def stream_events(self, messages, *, tools=None):
        self.calls.append([dict(m) for m in messages])
        deltas, response = self.turns[len(self.calls) - 1]
        for delta in deltas:
            yield TextDelta(delta)
        yield Completed(response)


async def test_astream_yields_text_as_it_arrives():
    provider = StreamingProvider([(["Hel", "lo"], CompletionResponse(content="Hello"))])
    agent = Agent(provider)

    chunks = [chunk async for chunk in agent.astream("hi")]

    assert chunks == ["Hel", "lo"]


async def test_a_streamed_run_still_dispatches_tools():
    from subagents.providers.base import ToolCall

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    provider = StreamingProvider(
        [
            (
                ["let me add that"],
                CompletionResponse(
                    content="let me add that",
                    tool_calls=[
                        ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})
                    ],
                ),
            ),
            (["The answer is ", "3"], CompletionResponse(content="The answer is 3")),
        ]
    )
    agent = Agent(provider, tools=[add])

    events = [event async for event in agent.astream_events("1 + 2?")]

    assert [e.text for e in events if isinstance(e, TextDelta)] == [
        "let me add that",
        "The answer is ",
        "3",
    ]
    final = events[-1]
    assert isinstance(final, Finished)
    assert final.state.output == "The answer is 3"
    assert final.state.answered
    assert any(
        m.get("role") == "tool" and m["content"] == "3" for m in final.state.messages
    )


def test_sync_stream_works_too():
    provider = StreamingProvider([(["a", "b"], CompletionResponse(content="ab"))])
    agent = Agent(provider)

    assert list(agent.stream("hi")) == ["a", "b"]


async def test_streaming_a_model_less_agent_just_finishes():
    events = [event async for event in Agent().astream_events("hi")]

    assert len(events) == 1
    assert isinstance(events[0], Finished)


async def test_structured_output_survives_streaming():
    from dataclasses import dataclass

    from subagents.agent.output import FINAL_TOOL
    from subagents.providers.base import ToolCall

    @dataclass
    class Weather:
        city: str

    provider = StreamingProvider(
        [
            (
                [],
                CompletionResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="c1", name=FINAL_TOOL, arguments={"city": "Oslo"})
                    ],
                ),
            )
        ]
    )
    agent = Agent(provider, output=Weather)

    events = [event async for event in agent.astream_events("where?")]

    assert events[-1].state.output == Weather(city="Oslo")
