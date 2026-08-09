from __future__ import annotations

from types import SimpleNamespace

import pytest

from providers import (
    CATALOG_ALIASES,
    MODEL_CATALOG,
    OFFICIAL_DEEPSEEK_BASE_URL,
    AnthropicAdapter,
    DeepSeekAdapter,
    GenerationFailedError,
    GenerationRequest,
    GenerationTarget,
    OpenAIResponsesAdapter,
    ProviderError,
    ProviderErrorCategory,
    RetryPolicy,
    default_adapter_factory,
    generate_with_fallback,
    normalize_provider_error,
    parse_retry_after,
    repair_malformed_json,
    resolve_api_key,
    resolve_generation_config,
)


def target(alias: str, key: str = "test-secret-key") -> GenerationTarget:
    return GenerationTarget(MODEL_CATALOG[alias], "session override", key)


def request() -> GenerationRequest:
    return GenerationRequest("system instructions", "document passage", 123)


def test_catalog_has_exact_curated_aliases_and_pinned_ids():
    assert CATALOG_ALIASES == (
        "Claude Sonnet 5",
        "Claude Haiku 4.5",
        "DeepSeek V4 Flash",
        "DeepSeek V4 Pro",
        "GPT-5.4 Mini",
        "GPT-5.6 Sol",
    )
    assert {name: spec.model_id for name, spec in MODEL_CATALOG.items()} == {
        "Claude Sonnet 5": "claude-sonnet-5",
        "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
        "DeepSeek V4 Flash": "deepseek-v4-flash",
        "DeepSeek V4 Pro": "deepseek-v4-pro",
        "GPT-5.4 Mini": "gpt-5.4-mini",
        "GPT-5.6 Sol": "gpt-5.6-sol",
    }
    assert "gpt-5.6" not in {spec.model_id for spec in MODEL_CATALOG.values()}
    assert all(spec.context_tokens > 0 and spec.max_output_tokens > 0 for spec in MODEL_CATALOG.values())


@pytest.mark.parametrize(
    ("provider", "key"),
    [("anthropic", "ANTHROPIC_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_credential_precedence_blanks_and_clearing(provider, key):
    resolution = resolve_api_key(provider, {key: "  session "}, {key: "secret"}, {key: "environment"})
    assert resolution.source == "session override"
    assert resolution.api_key == "session"

    resolution = resolve_api_key(provider, {key: "  "}, {key: " secret "}, {key: "environment"})
    assert resolution.source == "Streamlit secret"
    assert resolution.api_key == "secret"

    resolution = resolve_api_key(provider, {key: ""}, {key: "\t"}, {key: " env "})
    assert resolution.source == "environment"
    assert resolution.api_key == "env"

    resolution = resolve_api_key(provider, {}, {}, {})
    assert resolution.source == "missing"
    assert resolution.api_key is None


def test_credentials_only_read_canonical_provider_key_and_require_mappings():
    assert resolve_api_key("openai", {"DEEPSEEK_API_KEY": "wrong"}, {}, {}).is_available is False
    with pytest.raises(TypeError):
        resolve_api_key("openai", ["not", "a", "mapping"], {}, {})


def test_config_rejects_same_target_and_degrades_missing_secondary():
    primary_key = {"OPENAI_API_KEY": "primary-key"}
    with pytest.raises(ProviderError) as raised:
        resolve_generation_config("GPT-5.4 Mini", "GPT-5.4 Mini", fallback_enabled=True, session_overrides=primary_key)
    assert raised.value.category is ProviderErrorCategory.INVALID_REQUEST

    config = resolve_generation_config(
        "GPT-5.4 Mini",
        "DeepSeek V4 Flash",
        fallback_enabled=True,
        session_overrides=primary_key,
    )
    assert config.secondary is None
    assert "credential is missing" in config.warnings[0]
    assert config.primary.credential_source == "session override"


class Recorder:
    def __init__(self, response):
        self.response = response
        self.calls = []


class FakeAnthropicClient(Recorder):
    def __init__(self, response):
        super().__init__(response)
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeResponsesClient(Recorder):
    def __init__(self, response):
        super().__init__(response)
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeDeepSeekClient(Recorder):
    def __init__(self, response):
        super().__init__(response)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_anthropic_mapping_uses_top_level_system_user_messages_and_max_tokens():
    client = FakeAnthropicClient(SimpleNamespace(content=[SimpleNamespace(text="  result ")]))
    adapter = AnthropicAdapter(target("Claude Haiku 4.5"), client)
    assert adapter.generate(request()) == "result"
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert call["system"] == "system instructions"
    assert call["messages"] == [{"role": "user", "content": "document passage"}]
    assert call["max_tokens"] == 123
    assert call["temperature"] == 0.0


def test_openai_responses_mapping_and_unsupported_fields_are_omitted():
    client = FakeResponsesClient(SimpleNamespace(output_text="result"))
    adapter = OpenAIResponsesAdapter(target("GPT-5.4 Mini"), client)
    assert adapter.generate(request()) == "result"
    assert client.calls == [{
        "model": "gpt-5.4-mini",
        "instructions": "system instructions",
        "input": "document passage",
        "max_output_tokens": 123,
    }]
    assert "temperature" not in client.calls[0]
    assert "top_p" not in client.calls[0]
    assert "response_format" not in client.calls[0]
    assert "reasoning" not in client.calls[0]


def test_deepseek_mapping_uses_official_chat_shape_and_grounded_temperature():
    client = FakeDeepSeekClient(SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="result"))]))
    adapter = DeepSeekAdapter(target("DeepSeek V4 Flash"), client)
    assert adapter.generate(request()) == "result"
    assert OFFICIAL_DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert client.calls == [{
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "document passage"},
        ],
        "max_tokens": 123,
        "temperature": 0.0,
    }]


def test_all_gpt_catalog_entries_omit_temperature_for_responses_requests():
    for alias in ("GPT-5.4 Mini", "GPT-5.6 Sol"):
        client = FakeResponsesClient(SimpleNamespace(output_text="result"))
        assert OpenAIResponsesAdapter(target(alias), client).generate(request()) == "result"
        assert "temperature" not in client.calls[0]


def test_adapters_reject_empty_textual_results():
    adapter = OpenAIResponsesAdapter(target("GPT-5.4 Mini"), FakeResponsesClient(SimpleNamespace(output_text=" ")))
    with pytest.raises(ProviderError) as raised:
        adapter.generate(request())
    assert raised.value.category is ProviderErrorCategory.CONTENT


class FakeAdapter:
    def __init__(self, outcomes, calls, name):
        self.outcomes = outcomes
        self.calls = calls
        self.name = name

    def generate(self, received_request):
        self.calls.append((self.name, received_request))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def fake_factory(outcomes_by_alias, calls):
    def make(received_target):
        return FakeAdapter(outcomes_by_alias[received_target.spec.alias], calls, received_target.spec.alias)
    return make


def config_with_fallback():
    return resolve_generation_config(
        "GPT-5.4 Mini",
        "DeepSeek V4 Flash",
        fallback_enabled=True,
        session_overrides={"OPENAI_API_KEY": "openai", "DEEPSEEK_API_KEY": "deepseek"},
    )


def retryable(category):
    return ProviderError(category, "test failure")


def test_retry_then_primary_recovery_uses_one_delay_and_preserves_warnings():
    config = config_with_fallback()
    calls, sleeps = [], []
    result = generate_with_fallback(
        request(),
        config,
        fake_factory({"GPT-5.4 Mini": [retryable(ProviderErrorCategory.TIMEOUT), "recovered"], "DeepSeek V4 Flash": []}, calls),
        RetryPolicy(base_delay_seconds=1, jitter_ratio=0),
        sleeps.append,
        lambda: 0.5,
    )
    assert result.text == "recovered"
    assert result.used_fallback is False
    assert [entry[0] for entry in calls] == ["GPT-5.4 Mini", "GPT-5.4 Mini"]
    assert sleeps == [1]
    assert result.attempts[0].retried is True
    assert result.attempts[1].succeeded is True


def test_retryable_primary_exhaustion_uses_one_fallback_without_loop():
    config = config_with_fallback()
    calls = []
    result = generate_with_fallback(
        request(),
        config,
        fake_factory({
            "GPT-5.4 Mini": [retryable(ProviderErrorCategory.RATE_LIMIT), retryable(ProviderErrorCategory.RATE_LIMIT)],
            "DeepSeek V4 Flash": ["fallback result"],
        }, calls),
        RetryPolicy(jitter_ratio=0), lambda _: None, lambda: 0.5,
    )
    assert result.text == "fallback result"
    assert result.used_fallback is True
    assert [call[0] for call in calls] == ["GPT-5.4 Mini", "GPT-5.4 Mini", "DeepSeek V4 Flash"]


def test_permanent_failure_short_circuits_without_fallback():
    calls = []
    with pytest.raises(GenerationFailedError) as raised:
        generate_with_fallback(
            request(), config_with_fallback(),
            fake_factory({"GPT-5.4 Mini": [retryable(ProviderErrorCategory.AUTHENTICATION)], "DeepSeek V4 Flash": ["must not run"]}, calls),
            RetryPolicy(), lambda _: None, lambda: 0.5,
        )
    assert raised.value.category is ProviderErrorCategory.AUTHENTICATION
    assert [call[0] for call in calls] == ["GPT-5.4 Mini"]


def test_retry_after_is_clamped_and_invalid_values_are_ignored():
    assert parse_retry_after("120", 10) == 10
    assert parse_retry_after("2", 10) == 2
    assert parse_retry_after("-1", 10) is None
    assert parse_retry_after("not-a-delay", 10) is None


@pytest.mark.parametrize(
    ("exception", "category"),
    [
        (SimpleNamespace(status_code=401), ProviderErrorCategory.AUTHENTICATION),
        (SimpleNamespace(status_code=403), ProviderErrorCategory.PERMISSION),
        (SimpleNamespace(status_code=400), ProviderErrorCategory.INVALID_REQUEST),
        (SimpleNamespace(status_code=404), ProviderErrorCategory.MODEL),
        (SimpleNamespace(status_code=413), ProviderErrorCategory.INPUT_CONTEXT),
        (SimpleNamespace(status_code=429), ProviderErrorCategory.RATE_LIMIT),
        (SimpleNamespace(status_code=408), ProviderErrorCategory.TIMEOUT),
        (SimpleNamespace(status_code=409), ProviderErrorCategory.OVERLOAD),
        (SimpleNamespace(status_code=503), ProviderErrorCategory.SERVER),
        (SimpleNamespace(status_code=502, message="authentication failed upstream"), ProviderErrorCategory.SERVER),
        (SimpleNamespace(status_code=503, message="invalid api key from upstream"), ProviderErrorCategory.SERVER),
        (TimeoutError("timed out"), ProviderErrorCategory.TIMEOUT),
        (ConnectionError("connection refused"), ProviderErrorCategory.CONNECTION),
        (RuntimeError("content rejected by policy"), ProviderErrorCategory.CONTENT),
    ],
)
def test_error_categories_are_normalized(exception, category):
    # SimpleNamespace cannot be raised, so turn status cases into an exception.
    if isinstance(exception, SimpleNamespace):
        exception = type("StatusError", (Exception,), {"status_code": exception.status_code})(getattr(exception, "message", ""))
    assert normalize_provider_error(exception).category is category


def test_repair_is_pinned_one_call_and_contains_no_document_source_text():
    calls = []
    repair_target = target("DeepSeek V4 Flash")
    result = repair_malformed_json(
        "{broken", "AnswerSchema", repair_target,
        fake_factory({"DeepSeek V4 Flash": ["{\"fixed\": true}"]}, calls),
    )
    assert result == '{"fixed": true}'
    assert len(calls) == 1
    used_target, repair_request = calls[0]
    assert used_target == "DeepSeek V4 Flash"
    assert "{broken" in repair_request.user_text
    assert "AnswerSchema" in repair_request.user_text
    assert "document passage" not in repair_request.user_text


def test_redaction_keeps_keys_and_sdk_bodies_out_of_representations():
    key = "do-not-display-this-key"
    provider_response = "do-not-display-this-response"
    credential = resolve_api_key("openai", {"OPENAI_API_KEY": key}, {}, {})
    configured = resolve_generation_config("GPT-5.4 Mini", None, session_overrides={"OPENAI_API_KEY": key})
    error = normalize_provider_error(RuntimeError(provider_response))
    visible = repr(credential) + repr(configured) + str(error) + repr(error)
    assert key not in visible
    assert provider_response not in visible
    assert key not in repr(configured.primary)
    assert not hasattr(credential, "__dict__")
    assert not hasattr(configured.primary, "__dict__")
