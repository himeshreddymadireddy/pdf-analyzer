"""Provider-neutral generation boundary.

This module deliberately keeps credentials out of prompts, errors, and normal
representations.  It is the only place that knows the three SDK request shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
import random as _random
import time as _time
from typing import Any, Callable, Mapping, Protocol


CATALOG_REVIEWED_ON = "2026-08-09"
OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider_id: str
    model_id: str
    transport: str
    context_tokens: int
    max_output_tokens: int
    estimator_encoding: str
    supported_parameters: frozenset[str] = frozenset()
    minimum_temperature: float | None = None


# These are curated entries, intentionally not a discovery mechanism.  Dateless
# IDs are retained only where the provider documents that ID as pinned.
MODEL_CATALOG: dict[str, ModelSpec] = {
    "Claude Sonnet 5": ModelSpec("Claude Sonnet 5", "anthropic", "claude-sonnet-5", "anthropic_messages", 200_000, 64_000, "cl100k_base", frozenset({"temperature"}), 0.0),
    "Claude Haiku 4.5": ModelSpec("Claude Haiku 4.5", "anthropic", "claude-haiku-4-5-20251001", "anthropic_messages", 200_000, 64_000, "cl100k_base", frozenset({"temperature"}), 0.0),
    "DeepSeek V4 Flash": ModelSpec("DeepSeek V4 Flash", "deepseek", "deepseek-v4-flash", "deepseek_chat_completions", 128_000, 8_000, "cl100k_base", frozenset({"temperature", "top_p"}), 0.0),
    "DeepSeek V4 Pro": ModelSpec("DeepSeek V4 Pro", "deepseek", "deepseek-v4-pro", "deepseek_chat_completions", 128_000, 8_000, "cl100k_base", frozenset({"temperature", "top_p"}), 0.0),
    # GPT-5 Responses models reject non-default sampling parameters, so the
    # catalog explicitly omits temperature rather than relying on SDK defaults.
    "GPT-5.4 Mini": ModelSpec("GPT-5.4 Mini", "openai", "gpt-5.4-mini", "openai_responses", 128_000, 32_000, "o200k_base"),
    "GPT-5.6 Sol": ModelSpec("GPT-5.6 Sol", "openai", "gpt-5.6-sol", "openai_responses", 128_000, 32_000, "o200k_base"),
}

# Kept separate so UI code can present catalog aliases without being able to
# supply arbitrary provider IDs or model IDs.
CATALOG_ALIASES = tuple(MODEL_CATALOG)
_PROVIDER_KEY_NAMES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class ProviderErrorCategory(str, Enum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    MODEL = "model"
    INPUT_CONTEXT = "input_context"
    CONTENT = "content"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    OVERLOAD = "overload"
    SERVER = "server"
    UNKNOWN = "unknown"


_RETRYABLE_CATEGORIES = {
    ProviderErrorCategory.RATE_LIMIT,
    ProviderErrorCategory.TIMEOUT,
    ProviderErrorCategory.CONNECTION,
    ProviderErrorCategory.OVERLOAD,
    ProviderErrorCategory.SERVER,
}


class ProviderError(RuntimeError):
    """A safe error: it intentionally has no response body or credential."""

    def __init__(
        self,
        category: ProviderErrorCategory,
        message: str = "Generation request failed.",
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.category = category
        self.status_code = status_code
        self.retry_after = retry_after
        # Provider SDK messages can contain request/response content. Retain only
        # the normalized category in the externally printable exception.
        super().__init__(f"Provider request failed ({category.value}).")

    @property
    def retryable(self) -> bool:
        return self.category in _RETRYABLE_CATEGORIES


class CredentialUnavailableError(ProviderError):
    def __init__(self, provider_id: str) -> None:
        super().__init__(ProviderErrorCategory.AUTHENTICATION, "A required provider credential is missing.")
        self.provider_id = provider_id


class GenerationFailedError(ProviderError):
    """Raised after a bounded execution path has no usable textual result."""

    def __init__(self, error: ProviderError, attempts: tuple["GenerationAttempt", ...]) -> None:
        self.attempts = attempts
        super().__init__(error.category, str(error), status_code=error.status_code, retry_after=error.retry_after)


class CredentialResolution:
    """Credential metadata with an intentionally non-serializable raw key."""

    __slots__ = ("provider_id", "source", "_api_key")

    def __init__(self, provider_id: str, source: str, api_key: str | None = None) -> None:
        self.provider_id = provider_id
        self.source = source
        self._api_key = api_key

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def is_available(self) -> bool:
        return self._api_key is not None

    def __repr__(self) -> str:
        return f"CredentialResolution(provider_id={self.provider_id!r}, source={self.source!r})"


@dataclass(frozen=True)
class GenerationRequest:
    system_text: str
    user_text: str
    max_output_tokens: int
    grounded: bool = True


class GenerationTarget:
    """A model target whose raw credential cannot enter normal serialization."""

    __slots__ = ("spec", "credential_source", "_api_key")

    def __init__(self, spec: ModelSpec, credential_source: str, api_key: str) -> None:
        self.spec = spec
        self.credential_source = credential_source
        self._api_key = api_key

    @property
    def provider_id(self) -> str:
        return self.spec.provider_id

    @property
    def api_key(self) -> str:
        """For adapter construction only; never include this in UI/error data."""
        return self._api_key

    def __repr__(self) -> str:
        return f"GenerationTarget(spec={self.spec!r}, credential_source={self.credential_source!r})"


@dataclass(frozen=True)
class GenerationAttempt:
    target: str
    number: int
    category: ProviderErrorCategory | None
    retried: bool
    delay_seconds: float = 0.0
    succeeded: bool = False


@dataclass(frozen=True)
class GenerationResult:
    text: str
    target: GenerationTarget
    attempts: tuple[GenerationAttempt, ...]
    used_fallback: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedGenerationConfig:
    primary: GenerationTarget
    secondary_or_none: GenerationTarget | None
    warnings: tuple[str, ...] = ()

    @property
    def secondary(self) -> GenerationTarget | None:
        return self.secondary_or_none


@dataclass(frozen=True)
class RetryPolicy:
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    max_retry_after_seconds: float = 60.0
    jitter_ratio: float = 0.2


class ProviderAdapter(Protocol):
    def generate(self, request: GenerationRequest) -> str:
        ...


def get_model_spec(alias: str, provider_id: str | None = None) -> ModelSpec:
    """Return a curated model or reject unknown/custom and cross-provider IDs."""
    try:
        spec = MODEL_CATALOG[alias]
    except KeyError as exc:
        raise ProviderError(ProviderErrorCategory.MODEL, "The selected model is not in the curated catalog.") from exc
    if provider_id is not None and spec.provider_id != provider_id:
        raise ProviderError(ProviderErrorCategory.MODEL, "The selected model does not belong to that provider.")
    return spec


def _mapping_value(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _require_mapping(values: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be an ordinary mapping.")
    return values


def resolve_api_key(
    provider_id: str,
    session_overrides: Mapping[str, Any] | None,
    streamlit_values: Mapping[str, Any] | None,
    environ: Mapping[str, Any] | None,
) -> CredentialResolution:
    """Resolve a key from supplied mappings without touching Streamlit or dotenv.

    Whitespace-only values deliberately act as a cleared value and allow the next
    source to be considered.  Only the provider's own canonical key is read.
    """
    try:
        key_name = _PROVIDER_KEY_NAMES[provider_id]
    except KeyError as exc:
        raise ProviderError(ProviderErrorCategory.INVALID_REQUEST, "Unknown provider.") from exc
    overrides = _require_mapping(session_overrides, "session_overrides")
    secrets = _require_mapping(streamlit_values, "streamlit_values")
    environment = _require_mapping(environ, "environ")
    for source, values in (("session override", overrides), ("Streamlit secret", secrets), ("environment", environment)):
        value = _mapping_value(values, key_name)
        if value:
            return CredentialResolution(provider_id, source, value)
    return CredentialResolution(provider_id, "missing")


def _target_for(alias: str, session_overrides: Mapping[str, Any] | None, streamlit_values: Mapping[str, Any] | None, environ: Mapping[str, Any] | None) -> GenerationTarget | None:
    spec = get_model_spec(alias)
    credential = resolve_api_key(spec.provider_id, session_overrides, streamlit_values, environ)
    if not credential.is_available:
        return None
    return GenerationTarget(spec, credential.source, credential.api_key or "")


def resolve_generation_config(
    primary_alias: str,
    secondary_alias: str | None,
    *,
    fallback_enabled: bool = False,
    session_overrides: Mapping[str, Any] | None = None,
    streamlit_values: Mapping[str, Any] | None = None,
    environ: Mapping[str, Any] | None = None,
) -> ResolvedGenerationConfig:
    """Validate primary independently and degrade an unavailable fallback safely."""
    primary_spec = get_model_spec(primary_alias)
    primary = _target_for(primary_alias, session_overrides, streamlit_values, environ)
    if primary is None:
        raise CredentialUnavailableError(primary_spec.provider_id)

    warnings: list[str] = []
    secondary: GenerationTarget | None = None
    if fallback_enabled:
        if not secondary_alias:
            warnings.append("Fallback is enabled but no secondary model is selected.")
        else:
            secondary_spec = get_model_spec(secondary_alias)
            if (secondary_spec.provider_id, secondary_spec.model_id) == (primary.spec.provider_id, primary.spec.model_id):
                raise ProviderError(ProviderErrorCategory.INVALID_REQUEST, "Fallback must use a distinct provider/model target.")
            secondary = _target_for(secondary_alias, session_overrides, streamlit_values, environ)
            if secondary is None:
                warnings.append("Fallback is unavailable because its credential is missing; generation will use the primary only.")
    return ResolvedGenerationConfig(primary, secondary, tuple(warnings))


def _http_timeout() -> Any:
    try:
        import httpx
    except ImportError:  # SDK import below will produce the safe unavailable error.
        return None
    return httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)


def _grounded_parameters(spec: ModelSpec, request: GenerationRequest) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if request.grounded and "temperature" in spec.supported_parameters:
        params["temperature"] = 0.0 if spec.minimum_temperature is None else spec.minimum_temperature
    return params


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            text = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts).strip()
    return ""


def _require_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ProviderError(ProviderErrorCategory.CONTENT, "The provider returned no usable text.")
    return text.strip()


class AnthropicAdapter:
    def __init__(self, target: GenerationTarget, client: Any | None = None) -> None:
        if target.spec.provider_id != "anthropic" or target.spec.transport != "anthropic_messages":
            raise ProviderError(ProviderErrorCategory.MODEL, "Model transport does not match the Anthropic adapter.")
        self.target = target
        if client is None:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=target.api_key, timeout=_http_timeout(), max_retries=0)
            except ImportError as exc:
                raise ProviderError(ProviderErrorCategory.UNKNOWN, "Anthropic SDK is unavailable.") from exc
        self.client = client

    def generate(self, request: GenerationRequest) -> str:
        response = self.client.messages.create(
            model=self.target.spec.model_id,
            max_tokens=request.max_output_tokens,
            system=request.system_text,
            messages=[{"role": "user", "content": request.user_text}],
            **_grounded_parameters(self.target.spec, request),
        )
        return _require_text(_text_from_content(getattr(response, "content", None)))


class OpenAIResponsesAdapter:
    def __init__(self, target: GenerationTarget, client: Any | None = None) -> None:
        if target.spec.provider_id != "openai" or target.spec.transport != "openai_responses":
            raise ProviderError(ProviderErrorCategory.MODEL, "Model transport does not match the OpenAI adapter.")
        self.target = target
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=target.api_key, timeout=_http_timeout(), max_retries=0)
            except ImportError as exc:
                raise ProviderError(ProviderErrorCategory.UNKNOWN, "OpenAI SDK is unavailable.") from exc
        self.client = client

    def generate(self, request: GenerationRequest) -> str:
        response = self.client.responses.create(
            model=self.target.spec.model_id,
            instructions=request.system_text,
            input=request.user_text,
            max_output_tokens=request.max_output_tokens,
            **_grounded_parameters(self.target.spec, request),
        )
        return _require_text(getattr(response, "output_text", None))


class DeepSeekAdapter:
    def __init__(self, target: GenerationTarget, client: Any | None = None) -> None:
        if target.spec.provider_id != "deepseek" or target.spec.transport != "deepseek_chat_completions":
            raise ProviderError(ProviderErrorCategory.MODEL, "Model transport does not match the DeepSeek adapter.")
        self.target = target
        if client is None:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=OFFICIAL_DEEPSEEK_BASE_URL, api_key=target.api_key, timeout=_http_timeout(), max_retries=0)
            except ImportError as exc:
                raise ProviderError(ProviderErrorCategory.UNKNOWN, "OpenAI-compatible SDK is unavailable.") from exc
        self.client = client

    def generate(self, request: GenerationRequest) -> str:
        response = self.client.chat.completions.create(
            model=self.target.spec.model_id,
            messages=[
                {"role": "system", "content": request.system_text},
                {"role": "user", "content": request.user_text},
            ],
            max_tokens=request.max_output_tokens,
            **_grounded_parameters(self.target.spec, request),
        )
        choices = getattr(response, "choices", ())
        message = getattr(choices[0], "message", None) if choices else None
        return _require_text(getattr(message, "content", None))


def default_adapter_factory(target: GenerationTarget) -> ProviderAdapter:
    if target.spec.transport == "anthropic_messages":
        return AnthropicAdapter(target)
    if target.spec.transport == "openai_responses":
        return OpenAIResponsesAdapter(target)
    if target.spec.transport == "deepseek_chat_completions":
        return DeepSeekAdapter(target)
    raise ProviderError(ProviderErrorCategory.MODEL, "Unsupported catalog transport.")


def _status_from_exception(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _retry_after_from_exception(exc: BaseException, maximum: float = 60.0) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if isinstance(headers, Mapping) else None
    return parse_retry_after(value, maximum)


def parse_retry_after(value: Any, maximum: float = 60.0, *, now: datetime | None = None) -> float | None:
    """Return a finite, non-negative, clamped Retry-After delay or ``None``."""
    if not isinstance(maximum, (int, float)) or maximum < 0:
        raise ValueError("maximum must be non-negative")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = float(value.strip())
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                current = now or datetime.now(timezone.utc)
                seconds = (target - current).total_seconds()
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
    else:
        return None
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return None
    return min(seconds, float(maximum))


def normalize_provider_error(exc: BaseException, retry_after_maximum: float = 60.0) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    status = _status_from_exception(exc)
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    marker = f"{name} {message}"
    # Exact HTTP statuses are authoritative. In particular, a gateway/server
    # response might include an upstream authentication string but remains
    # retryable server failure rather than a permanent credential failure.
    if status == 401:
        category = ProviderErrorCategory.AUTHENTICATION
    elif status == 403:
        category = ProviderErrorCategory.PERMISSION
    elif status == 404:
        category = ProviderErrorCategory.MODEL
    elif status == 408:
        category = ProviderErrorCategory.TIMEOUT
    elif status == 409:
        category = ProviderErrorCategory.OVERLOAD
    elif status == 429:
        category = ProviderErrorCategory.RATE_LIMIT
    elif status is not None and 500 <= status <= 599:
        category = ProviderErrorCategory.SERVER
    elif status in (413, 414, 422):
        category = ProviderErrorCategory.INPUT_CONTEXT
    # A 400 can still identify a permanent input/context or content issue.
    elif "content" in marker and ("reject" in marker or "filter" in marker or "policy" in marker):
        category = ProviderErrorCategory.CONTENT
    elif "context" in marker or "input too" in marker or "too many token" in marker:
        category = ProviderErrorCategory.INPUT_CONTEXT
    elif "authentication" in marker or "api key" in marker:
        category = ProviderErrorCategory.AUTHENTICATION
    elif "permission" in marker or "forbidden" in marker:
        category = ProviderErrorCategory.PERMISSION
    elif "notfound" in marker or "model not found" in marker:
        category = ProviderErrorCategory.MODEL
    elif "timeout" in marker or "timed out" in marker:
        category = ProviderErrorCategory.TIMEOUT
    elif "overload" in marker or "overloaded" in marker:
        category = ProviderErrorCategory.OVERLOAD
    elif "ratelimit" in marker or "rate limit" in marker:
        category = ProviderErrorCategory.RATE_LIMIT
    elif "connection" in marker or "connecterror" in marker or "network" in marker:
        category = ProviderErrorCategory.CONNECTION
    elif status == 400 or "badrequest" in marker or "invalid request" in marker:
        category = ProviderErrorCategory.INVALID_REQUEST
    else:
        category = ProviderErrorCategory.UNKNOWN
    return ProviderError(category, "The provider request failed.", status_code=status, retry_after=_retry_after_from_exception(exc, retry_after_maximum))


def _backoff(error: ProviderError, retry_number: int, policy: RetryPolicy, random_value: Callable[[], float]) -> float:
    retry_after = error.retry_after
    if retry_after is not None:
        return min(retry_after, policy.max_retry_after_seconds)
    unjittered = min(policy.max_delay_seconds, policy.base_delay_seconds * (2 ** retry_number))
    random_unit = min(1.0, max(0.0, float(random_value())))
    return max(0.0, unjittered * (1.0 + ((random_unit * 2.0 - 1.0) * policy.jitter_ratio)))


def _attempt_target(
    request: GenerationRequest,
    target: GenerationTarget,
    adapter_factory: Callable[[GenerationTarget], ProviderAdapter],
    policy: RetryPolicy,
    sleep: Callable[[float], None],
    random_value: Callable[[], float],
) -> tuple[str | None, tuple[GenerationAttempt, ...], ProviderError | None]:
    attempts: list[GenerationAttempt] = []
    for number in range(2):  # One initial attempt and one service-owned retry.
        try:
            text = adapter_factory(target).generate(request)
            attempts.append(GenerationAttempt(target.spec.alias, number + 1, None, False, succeeded=True))
            return text, tuple(attempts), None
        except Exception as exc:  # Normalized immediately; never retain SDK response objects.
            error = normalize_provider_error(exc, policy.max_retry_after_seconds)
            will_retry = error.retryable and number == 0
            delay = _backoff(error, number, policy, random_value) if will_retry else 0.0
            attempts.append(GenerationAttempt(target.spec.alias, number + 1, error.category, will_retry, delay))
            if not will_retry:
                return None, tuple(attempts), error
            sleep(delay)
    raise AssertionError("bounded retry loop must return")


def generate_with_fallback(
    request: GenerationRequest,
    config: ResolvedGenerationConfig,
    adapter_factory: Callable[[GenerationTarget], ProviderAdapter] = default_adapter_factory,
    retry_policy: RetryPolicy = RetryPolicy(),
    sleep: Callable[[float], None] = _time.sleep,
    random: Callable[[], float] = _random.random,
) -> GenerationResult:
    """Run primary then, only after retryable exhaustion, one distinct fallback."""
    primary_text, primary_attempts, primary_error = _attempt_target(request, config.primary, adapter_factory, retry_policy, sleep, random)
    if primary_text is not None:
        return GenerationResult(primary_text, config.primary, primary_attempts, False, config.warnings)
    assert primary_error is not None
    if not primary_error.retryable or config.secondary_or_none is None:
        raise GenerationFailedError(primary_error, primary_attempts)

    secondary_text, secondary_attempts, secondary_error = _attempt_target(request, config.secondary_or_none, adapter_factory, retry_policy, sleep, random)
    all_attempts = primary_attempts + secondary_attempts
    if secondary_text is not None:
        return GenerationResult(secondary_text, config.secondary_or_none, all_attempts, True, config.warnings)
    assert secondary_error is not None
    raise GenerationFailedError(secondary_error, all_attempts)


def repair_malformed_json(
    malformed_text: str,
    schema_name: str,
    successful_target: GenerationTarget,
    adapter_factory: Callable[[GenerationTarget], ProviderAdapter] = default_adapter_factory,
) -> str:
    """Perform exactly one same-target repair call without fallback or retry.

    The request deliberately includes only the malformed model output and schema
    instruction.  Source document text is neither accepted nor reconstructed.
    """
    if not isinstance(malformed_text, str) or not malformed_text.strip():
        raise ProviderError(ProviderErrorCategory.INVALID_REQUEST, "Malformed JSON text is required for repair.")
    if not isinstance(schema_name, str) or not schema_name.strip():
        raise ProviderError(ProviderErrorCategory.INVALID_REQUEST, "A schema name is required for repair.")
    request = GenerationRequest(
        system_text="Return only valid JSON matching the requested schema. Do not add commentary.",
        user_text=f"Schema: {schema_name}\n\nMalformed JSON to repair:\n{malformed_text}",
        max_output_tokens=min(successful_target.spec.max_output_tokens, 4096),
        grounded=True,
    )
    try:
        return _require_text(adapter_factory(successful_target).generate(request))
    except Exception as exc:
        raise normalize_provider_error(exc) from None
