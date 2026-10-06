import pytest

from agentkit.llm import ConfigError, FallbackLLM, OpenAICompatLLM, from_env


def test_single_provider_default_model():
    llm = from_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"})
    assert isinstance(llm, OpenAICompatLLM) and llm.label == "gemini:gemini-2.5-flash"


def test_chain_with_model_overrides_splits_on_first_colon_only():
    env = {"LLM_PROVIDERS": "openrouter:qwen/qwen3-coder:free, gemini", "OPENROUTER_API_KEY": "a", "GEMINI_API_KEY": "b"}
    llm = from_env(env)
    assert isinstance(llm, FallbackLLM)
    assert [x.label for x in llm.llms] == ["openrouter:qwen/qwen3-coder:free", "gemini:gemini-2.5-flash"]


def test_providers_without_keys_are_skipped_but_others_work():
    env = {"LLM_PROVIDERS": "groq,gemini", "GEMINI_API_KEY": "b"}
    llm = from_env(env)
    assert isinstance(llm, OpenAICompatLLM) and llm.label.startswith("gemini")


def test_no_usable_provider_explains_what_to_set():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        from_env({})


def test_unknown_provider():
    with pytest.raises(ConfigError, match="Unknown provider"):
        from_env({"LLM_PROVIDER": "skynet"})


def test_ollama_needs_no_key():
    llm = from_env({"LLM_PROVIDER": "ollama:qwen3:14b"})
    assert llm.label == "ollama:qwen3:14b"


def test_custom_endpoint():
    env = {"LLM_PROVIDER": "custom", "LLM_BASE_URL": "http://x/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m"}
    assert from_env(env).label == "custom:m"
    with pytest.raises(ConfigError):
        from_env({"LLM_PROVIDER": "custom"})


def test_temperature_is_optional():
    assert from_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}).temperature is None
    assert from_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k", "LLM_TEMPERATURE": "0.2"}).temperature == 0.2


def test_extra_body_is_per_provider():
    env = {"LLM_PROVIDERS": "ollama:qwen3-vl:8b,gemini", "GEMINI_API_KEY": "k", "LLM_EXTRA_BODY_OLLAMA": '{"think": false}'}
    llm = from_env(env)
    assert llm.llms[0].extra_body == {"think": False}
    assert llm.llms[1].extra_body is None


def test_extra_body_must_be_a_json_object():
    with pytest.raises(ConfigError, match="LLM_EXTRA_BODY_OLLAMA"):
        from_env({"LLM_PROVIDER": "ollama", "LLM_EXTRA_BODY_OLLAMA": "{not json"})
    with pytest.raises(ConfigError, match="JSON object"):
        from_env({"LLM_PROVIDER": "ollama", "LLM_EXTRA_BODY_OLLAMA": "[1]"})


def test_extra_body_is_sent_with_the_request():
    sent = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            raise __import__("openai").APIConnectionError(request=__import__("httpx").Request("POST", "http://x"))

    llm = OpenAICompatLLM(base_url="http://x/v1", api_key="k", model="m", extra_body={"think": False})
    llm.client.chat.completions = FakeCompletions()
    with pytest.raises(Exception):
        llm.chat([{"role": "user", "content": "hi"}])
    assert sent["extra_body"] == {"think": False}
