from unittest.mock import AsyncMock

from deepharness.providers.gateways import Groq, LlamaCpp, Ollama
from deepharness.providers.openai import _build_payload


def test_groq_uses_its_own_base_url_and_env_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")

    provider = Groq("llama-3.3-70b-versatile")

    assert (
        str(provider._http._async_client.base_url) == "https://api.groq.com/openai/v1/"
    )
    assert (
        provider._http._async_client.headers["authorization"] == "Bearer test-groq-key"
    )


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-env")

    provider = Groq("llama-3.3-70b-versatile", api_key="explicit-key")

    assert (
        provider._http._async_client.headers["authorization"] == "Bearer explicit-key"
    )


def test_local_gateway_has_no_env_lookup_and_no_auth_header(monkeypatch):
    """A local server takes no credential, and "Bearer " with an empty value is
    an illegal header value that httpx refuses to send at request time."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    provider = Ollama("llama3")

    assert str(provider._http._async_client.base_url) == "http://localhost:11434/v1/"
    assert "authorization" not in provider._http._async_client.headers


def test_base_url_param_overrides_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")

    provider = Groq("llama-3.3-70b-versatile", base_url="https://custom.internal/v1")

    assert str(provider._http._async_client.base_url) == "https://custom.internal/v1/"


def test_temperature_is_forwarded_to_the_request():
    client = AsyncMock()
    provider = Groq(
        "llama-3.3-70b-versatile", api_key="x", temperature=0.2, client=client
    )

    payload = _build_payload(
        provider._model,
        [{"role": "user", "content": "hi"}],
        None,
        provider._temperature,
    )

    assert payload.temperature == 0.2


def test_temperature_omitted_by_default():
    payload = _build_payload("model", [{"role": "user", "content": "hi"}], None, None)

    assert "temperature" not in payload.to_json()


def test_llamacpp_defaults_to_the_llama_server_port(monkeypatch):
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)

    provider = LlamaCpp("qwen3-4b")

    assert str(provider._http._async_client.base_url) == "http://localhost:8080/v1/"
    assert "authorization" not in provider._http._async_client.headers


def test_llamacpp_accepts_a_remote_base_url():
    """llama-server behind a tunnel or reverse proxy is the same wire format."""
    provider = LlamaCpp("qwen3-4b", base_url="https://llm.example/v1")

    assert str(provider._http._async_client.base_url) == "https://llm.example/v1/"
