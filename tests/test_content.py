"""Content blocks: images and files in, reasoning out, per vendor."""

import base64

import pytest

from deepharness.agent import Agent, Message
from deepharness.errors import ConfigurationError
from deepharness.providers.anthropic import _to_anthropic_messages
from deepharness.providers.base import (
    LLM,
    CompletionResponse,
    TextDelta,
    ThinkingDelta,
)
from deepharness.providers.content import (
    Document,
    Image,
    Text,
    Thinking,
    merge,
    parse,
    text_of,
    to_wire,
)
from deepharness.providers.gemini import _to_gemini_parts
from deepharness.providers.openai import _to_openai_messages


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG\r\n")
    return path


def test_an_image_needs_data_or_a_url_but_not_both():
    with pytest.raises(ConfigurationError):
        Image()
    with pytest.raises(ConfigurationError):
        Image(data="x", url="http://example.com/a.png")


def test_an_image_from_a_path_carries_its_media_type(png):
    image = Image.from_path(png)

    assert image.media_type == "image/png"
    assert base64.b64decode(image.data) == b"\x89PNG\r\n"
    assert image.data_url.startswith("data:image/png;base64,")


def test_an_unknown_suffix_is_refused_rather_than_guessed(tmp_path):
    path = tmp_path / "thing.bin"
    path.write_bytes(b"x")

    with pytest.raises(ConfigurationError, match="media type"):
        Image.from_path(path)


def test_content_round_trips_through_its_wire_form():
    blocks = [Text("look"), Image(url="http://example.com/a.png")]

    assert parse(to_wire(blocks)) == blocks


def test_plain_text_stays_a_plain_string_on_the_wire():
    assert to_wire("hello") == "hello"
    assert to_wire([Text("hello")]) == "hello"


def test_an_unknown_block_type_is_an_error():
    with pytest.raises(ConfigurationError, match="unknown content block"):
        parse([{"type": "video", "data": "x"}])


def test_text_of_leaves_thinking_out():
    assert text_of([Thinking("hmm"), Text("answer")]) == "answer"


def test_merge_joins_runs_of_the_same_kind():
    merged = merge([Text("a"), Text("b"), Thinking("c"), Text("d")])

    assert merged == [Text("ab"), Thinking("c"), Text("d")]


def test_merge_keeps_thinking_with_different_signatures_apart():
    blocks = [Thinking("a", signature="s1"), Thinking("b", signature="s2")]

    assert merge(blocks) == blocks


def test_a_message_takes_one_block_or_a_list(png):
    assert Message.human(Image.from_path(png)).content == [Image.from_path(png)]
    assert Message.human("hi").content == "hi"


def test_a_message_exposes_its_text_whatever_the_blocks(png):
    message = Message.human([Text("look at this"), Image.from_path(png)])

    assert message.text == "look at this"
    assert isinstance(message.to_dict()["content"], list)


def test_a_completion_keeps_content_and_blocks_in_step():
    from_text = CompletionResponse(content="hi")
    from_blocks = CompletionResponse(blocks=[Thinking("hmm"), Text("hi")])

    assert from_text.blocks == [Text("hi")]
    assert from_blocks.content == "hi"
    assert from_blocks.thinking == "hmm"
    assert from_text.thinking == ""


def test_openai_sends_an_image_as_an_image_url_part(png):
    messages = [Message.human([Text("what is this?"), Image.from_path(png)]).to_dict()]

    converted = _to_openai_messages(messages)

    assert converted[0]["content"][0] == {"type": "text", "text": "what is this?"}
    assert converted[0]["content"][1]["image_url"]["url"].startswith("data:image/png")


def test_openai_sends_a_document_as_a_file_part(tmp_path):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    messages = [Message.human([Document.from_path(pdf)]).to_dict()]

    part = _to_openai_messages(messages)[0]["content"][0]

    assert part["type"] == "file"
    assert part["file"]["filename"] == "report.pdf"


def test_openai_keeps_plain_text_as_a_string():
    converted = _to_openai_messages([Message.human("hi").to_dict()])

    assert converted[0]["content"] == "hi"


def test_anthropic_sends_an_image_as_a_base64_source(png):
    messages = [Message.human([Image.from_path(png)]).to_dict()]

    _, converted = _to_anthropic_messages(messages)

    assert converted[0]["content"][0]["source"]["type"] == "base64"
    assert converted[0]["content"][0]["source"]["media_type"] == "image/png"


def test_anthropic_sends_an_image_url_as_a_url_source():
    messages = [Message.human([Image(url="http://example.com/a.png")]).to_dict()]

    _, converted = _to_anthropic_messages(messages)

    assert converted[0]["content"][0]["source"] == {
        "type": "url",
        "url": "http://example.com/a.png",
    }


def test_anthropic_replays_signed_thinking_with_the_turn():
    messages = [
        Message.ai(
            [Thinking("step one", signature="sig"), Text("done")],
            tool_calls=[{"id": "1", "name": "t", "arguments": {}}],
        ).to_dict()
    ]

    _, converted = _to_anthropic_messages(messages)

    assert converted[0]["content"][0] == {
        "type": "thinking",
        "thinking": "step one",
        "signature": "sig",
    }
    assert converted[0]["content"][-1]["type"] == "tool_use"


def test_anthropic_leaves_unsigned_thinking_out():
    messages = [Message.ai([Thinking("unsigned"), Text("done")]).to_dict()]

    _, converted = _to_anthropic_messages(messages)

    assert converted[0]["content"] == [{"type": "text", "text": "done"}]


def test_gemini_sends_an_image_inline_and_drops_thinking(png):
    parts = _to_gemini_parts([Thinking("hmm"), Text("look"), Image.from_path(png)])

    assert parts[0] == {"text": "look"}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"


def test_gemini_refuses_an_image_url_it_cannot_send():
    with pytest.raises(ConfigurationError, match="inline image data"):
        _to_gemini_parts([Image(url="http://example.com/a.png")])


class Thinker(LLM):
    """A model that reports reasoning before it answers.

    Streaming comes from LLM itself, which is the point of the test: a backend
    that cannot stream still reports thinking and prose as separate deltas.
    """

    def __init__(self, responses):
        self.responses = responses
        self.sent = []

    async def agenerate(self, messages, *, tools=None):
        return self.generate(messages, tools=tools)

    def generate(self, messages, *, tools=None):
        self.sent.append([dict(m) for m in messages])
        return self.responses[len(self.sent) - 1]


def test_a_run_keeps_the_model_s_thinking_in_the_transcript():
    model = Thinker(
        [CompletionResponse(blocks=[Thinking("hmm", signature="s"), Text("42")])]
    )
    agent = Agent(model)

    state = agent.run("what is it?")

    assert state.output == "42"
    assert state.messages[-1]["content"] == [
        {"type": "thinking", "text": "hmm", "signature": "s"},
        {"type": "text", "text": "42"},
    ]


def test_streaming_a_run_reports_thinking_separately_from_the_answer():
    model = Thinker([CompletionResponse(blocks=[Thinking("hmm"), Text("42")])])

    events = list(Agent(model).stream_events("what is it?"))

    assert [e for e in events if isinstance(e, ThinkingDelta)] == [ThinkingDelta("hmm")]
    assert [e for e in events if isinstance(e, TextDelta)] == [TextDelta("42")]


def test_streaming_text_alone_leaves_thinking_out():
    model = Thinker([CompletionResponse(blocks=[Thinking("hmm"), Text("42")])])

    assert list(Agent(model).stream("what is it?")) == ["42"]
