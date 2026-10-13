"""LLM layer — one interface, three pluggable providers.

``complete_json(system, user, schema, agent_name) -> (BaseModel, meta)``

Providers (selected by ``LLM_PROVIDER`` env var):

* ``gemini`` (default): ``google-genai`` SDK in Vertex AI mode, structured
  output with a JSON schema derived from the pydantic model, temperature 0.
* ``openai_compatible``: plain ``httpx`` POST to
  ``${OPENAI_BASE_URL}/chat/completions`` with
  ``response_format={"type": "json_object"}``.
* ``mock``: deterministic outputs keyed by agent name, built from the
  evidence pack so they always cite real IDs; used by all tests.

Retries with exponential backoff and jitter on transport failures
(disconnects, connection resets, timeouts), HTTP 429 and 5xx — at least
5 attempts spanning about 30 seconds.  Permanent failures (400 bad
request, 401/403 auth, schema/validation errors) fail immediately.  A
per-case token budget (``CASE_TOKEN_BUDGET``, default 60000) accumulates
across calls; when exceeded a ``TokenBudgetExceeded`` error is raised so
the orchestrator can escalate.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "gemini")


def _model() -> str:
    return os.environ.get("MODEL", DEFAULT_MODEL)


def _case_token_budget() -> int:
    return int(os.environ.get("CASE_TOKEN_BUDGET", "60000"))


def _gemini_thinking() -> str:
    """Return the thinking level: ``low`` (default), ``medium``, ``high``, ``off``."""
    return os.environ.get("GEMINI_THINKING", "low").lower()


# Mapping from env level to thinking_budget tokens accepted by google-genai.
# ``low`` keeps latency manageable for one-line structured calls.
_THINKING_BUDGET_MAP = {
    "off": 0,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
}


# ---------------------------------------------------------------------------
# Meta + errors
# ---------------------------------------------------------------------------


class LLMMeta(BaseModel):
    """Per-call metadata returned alongside the parsed model."""

    provider: str
    model: str
    agent_name: str
    tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = {}


class TokenBudgetExceeded(Exception):
    """Raised when cumulative tokens for the case exceed the budget."""

    def __init__(self, spent: int, budget: int):
        self.spent = spent
        self.budget = budget
        super().__init__(f"token budget exceeded: {spent} > {budget}")


class LLMError(Exception):
    """Raised when the LLM provider fails after retries (retriable by default).

    The retry loop in ``_retry`` treats a bare ``LLMError`` as retriable.
    Raise ``PermanentLLMError`` instead when the error cannot be fixed by
    retrying (400 bad request, 401/403 auth, schema/validation errors).
    """


class PermanentLLMError(LLMError):
    """An LLM error that a retry cannot fix.

    Raised for HTTP 400, 401, 403, and for schema/validation failures.
    ``_retry`` re-raises these immediately without sleeping or retrying.
    """


# ---------------------------------------------------------------------------
# Token budget tracker (per case)
# ---------------------------------------------------------------------------


class CaseBudget:
    """Accumulates token spend for one case; raises when exceeded."""

    def __init__(self, budget: int | None = None):
        self.budget = budget if budget is not None else _case_token_budget()
        self.spent = 0

    def charge(self, tokens: int) -> None:
        self.spent += tokens
        if self.spent > self.budget:
            raise TokenBudgetExceeded(self.spent, self.budget)


# Module-level singleton for the active case.  The orchestrator calls
# ``set_case_budget`` when it starts a case; individual ``complete_json``
# calls charge against it.
_active_budget: CaseBudget | None = None


def set_case_budget(budget: CaseBudget | None) -> None:
    global _active_budget
    _active_budget = budget


def _charge(tokens: int) -> None:
    if _active_budget is not None and tokens > 0:
        _active_budget.charge(tokens)


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------


def _pydantic_to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a pydantic v2 model class to a JSON-schema dict."""
    return model.model_json_schema()


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

# At least 5 attempts; total backoff about 30 s with exponential growth
# plus up to 25% jitter per delay.  ``_INITIAL_BACKOFF`` * 2^(n-1) with
# jitter gives roughly: 1, 2, 4, 8, 16 → ~31 s before the final attempt.
_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF = 1.0  # seconds
_MAX_BACKOFF = 16.0  # cap any single delay


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter for *attempt* (0-based)."""
    base = min(_INITIAL_BACKOFF * (2 ** attempt), _MAX_BACKOFF)
    # Full jitter: uniform random in [0, base], capped.
    return random.uniform(0, base)


def _is_retriable(exc: Exception) -> bool:
    """Return True if *exc* represents a transient error worth retrying."""
    if isinstance(exc, PermanentLLMError):
        return False
    if isinstance(exc, LLMError):
        # Bare LLMError from the provider wrappers is retriable (transport
        # errors, 429, 5xx).  The provider wrapper decides what to raise;
        # anything it wraps in a plain LLMError is considered transient.
        return True
    # httpx transport errors raised directly (not wrapped in LLMError).
    if isinstance(exc, (
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        ConnectionError,
        TimeoutError,
    )):
        return True
    return False


def _retry(fn, *args, **kwargs):
    """Retry *fn* on transient errors up to ``_MAX_ATTEMPTS`` times.

    Permanent failures (``PermanentLLMError``) are re-raised immediately.
    Retries use exponential backoff with full jitter; the total sleep
    budget across all attempts is about 30 seconds.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retriable(exc):
                raise
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s",
                    _MAX_ATTEMPTS,
                    exc,
                )
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def complete_json(
    system: str,
    user: str,
    schema: type[T],
    agent_name: str,
    *,
    evidence_pack: Any | None = None,
    policy_result: Any | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[T, LLMMeta]:
    """Call the configured LLM provider and parse the response into *schema*.

    Returns ``(parsed_model, meta)`` where *meta* carries tokens and latency.

    *evidence_pack* and *policy_result* are passed to the mock provider so it
    can build deterministic outputs that cite real evidence IDs.
    """
    provider = _provider()

    if provider == "mock":
        return _complete_mock(
            system, user, schema, agent_name,
            evidence_pack=evidence_pack,
            policy_result=policy_result,
            context=context or {},
        )
    if provider == "gemini":
        return _retry(
            _complete_gemini, system, user, schema, agent_name,
        )
    if provider == "openai_compatible":
        return _retry(
            _complete_openai, system, user, schema, agent_name,
        )

    raise LLMError(f"unknown LLM_PROVIDER: {provider!r}")


# ---------------------------------------------------------------------------
# Gemini provider (Vertex AI mode)
# ---------------------------------------------------------------------------

# Status codes that are permanent — a retry cannot fix them.
_PERMANENT_STATUS_CODES = {400, 401, 403}
# Status codes that are transient — retrying may succeed.
_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _classify_gemini_error(exc: Exception) -> LLMError:
    """Classify a google-genai SDK exception as retriable or permanent.

    The SDK may raise:
    * ``google.genai.errors.ClientError`` (and similar) carrying a
      ``response`` with a status code;
    * httpx transport errors (``RemoteProtocolError``, ``ConnectError``,
      ``ReadTimeout``, etc.) when the connection drops;
    * Generic ``Exception`` wrappers with the original error in the
      message.

    We inspect the exception's attributes and message to decide.
    """
    # Check for an HTTP status code on the exception or its cause.
    status = _extract_status(exc)
    if status is not None:
        if status in _PERMANENT_STATUS_CODES:
            return PermanentLLMError(f"gemini API error: status {status}: {exc}")
        if status in _RETRIABLE_STATUS_CODES:
            return LLMError(f"gemini API error: status {status}: {exc}")
        # Unknown status — treat as retriable (safer default for a
        # transient-looking failure).
        return LLMError(f"gemini API error: status {status}: {exc}")

    # Transport errors (disconnect, connection reset, timeout).
    cause = exc.__cause__ or exc
    if isinstance(cause, (
        httpx.RemoteProtocolError,
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        ConnectionError,
        TimeoutError,
    )):
        return LLMError(f"gemini transport error: {exc}")

    # If the message looks like a known permanent condition, mark it so.
    msg = str(exc).lower()
    permanent_markers = [
        "400 ", "bad request", "invalid argument",
        "401 ", "unauthenticated",
        "403 ", "permission denied",
    ]
    for marker in permanent_markers:
        if marker in msg:
            return PermanentLLMError(f"gemini API error: {exc}")

    # Default: treat as retriable (disconnects, 500s, timeouts).
    return LLMError(f"gemini API error: {exc}")


def _extract_status(exc: Exception) -> int | None:
    """Try to pull an HTTP status code out of an SDK exception."""
    for obj in (exc, exc.__cause__):
        if obj is None:
            continue
        # google-genai errors may carry a ``response`` (httpx.Response).
        resp = getattr(obj, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            if isinstance(code, int):
                return code
        # Some SDK errors expose ``code`` or ``status``.
        for attr in ("code", "status"):
            val = getattr(obj, attr, None)
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
        # The httpx.Response may be nested under ``cause``.
        cause = getattr(obj, "__cause__", None)
        if cause is not None:
            resp = getattr(cause, "response", None)
            if resp is not None:
                code = getattr(resp, "status_code", None)
                if isinstance(code, int):
                    return code
    return None


def _complete_gemini(
    system: str,
    user: str,
    schema: type[T],
    agent_name: str,
) -> tuple[T, LLMMeta]:
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
    except ImportError as exc:
        raise LLMError("google-genai SDK not installed") from exc

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    model = _model()
    thinking_level = _gemini_thinking()

    client = genai.Client(
        vertexai=use_vertex,
        project=project,
        location=location,
    )

    json_schema = _pydantic_to_json_schema(schema)

    # Build thinking config; google-genai accepts a ThinkingConfigDict.
    thinking_budget = _THINKING_BUDGET_MAP.get(thinking_level, 1024)
    thinking_cfg: dict[str, Any] | None = None
    if thinking_budget > 0:
        thinking_cfg = {"thinking_budget": thinking_budget}
    else:
        # Explicitly disable thinking.
        thinking_cfg = {"thinking_budget": 0}

    config = GenerateContentConfig(
        system_instruction=system,
        temperature=0.0,
        response_mime_type="application/json",
        response_json_schema=json_schema,
        thinking_config=thinking_cfg,
    )

    start = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=config,
        )
    except Exception as exc:
        # Distinguish retriable transport errors / status codes from
        # permanent failures (400, 401, 403).
        raise _classify_gemini_error(exc) from exc
    latency_ms = int((time.monotonic() - start) * 1000)

    raw_text = response.text or ""
    parsed_dict = _parse_json(raw_text)
    usage = getattr(response, "usage_metadata", None)
    tokens = 0
    if usage is not None:
        tokens = (
            getattr(usage, "total_token_count", None)
            or getattr(usage, "total_tokens", None)
            or 0
        )

    _charge(tokens)

    try:
        parsed = schema.model_validate(parsed_dict)
    except ValidationError as exc:
        raise PermanentLLMError(f"gemini response failed validation: {exc}") from exc

    meta = LLMMeta(
        provider="gemini",
        model=model,
        agent_name=agent_name,
        tokens=tokens,
        latency_ms=latency_ms,
        raw={"usage": str(usage) if usage else ""},
    )
    return parsed, meta


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


def _complete_openai(
    system: str,
    user: str,
    schema: type[T],
    agent_name: str,
) -> tuple[T, LLMMeta]:
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = _model()

    if not base_url:
        raise LLMError("OPENAI_BASE_URL is not set")

    url = base_url.rstrip("/") + "/chat/completions"
    json_schema = _pydantic_to_json_schema(schema)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        # Many OpenAI-compatible servers accept a schema hint.
        "extra_body": {"json_schema": json_schema},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.monotonic()
    try:
        resp = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        # Transport errors are retriable.
        raise LLMError(f"openai_compatible transport error: {exc}") from exc
    latency_ms = int((time.monotonic() - start) * 1000)

    if resp.status_code == 429 or resp.status_code >= 500:
        raise LLMError(
            f"openai_compatible retriable status {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code in _PERMANENT_STATUS_CODES:
        raise PermanentLLMError(
            f"openai_compatible status {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 400:
        raise LLMError(
            f"openai_compatible status {resp.status_code}: {resp.text[:200]}"
        )

    body = resp.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed_dict = _parse_json(content)

    usage = body.get("usage", {})
    tokens = usage.get("total_tokens", 0)

    _charge(tokens)

    try:
        parsed = schema.model_validate(parsed_dict)
    except ValidationError as exc:
        raise PermanentLLMError(f"openai_compatible response failed validation: {exc}") from exc

    meta = LLMMeta(
        provider="openai_compatible",
        model=model,
        agent_name=agent_name,
        tokens=tokens,
        latency_ms=latency_ms,
        raw={"usage": usage},
    )
    return parsed, meta


# ---------------------------------------------------------------------------
# Mock provider — deterministic, cites real evidence IDs
# ---------------------------------------------------------------------------


def _complete_mock(
    system: str,
    user: str,
    schema: type[T],
    agent_name: str,
    *,
    evidence_pack: Any | None = None,
    policy_result: Any | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[T, LLMMeta]:
    data = _mock_payload(
        agent_name,
        schema,
        evidence_pack=evidence_pack,
        policy_result=policy_result,
        context=context or {},
    )
    try:
        parsed = schema.model_validate(data)
    except ValidationError as exc:
        raise PermanentLLMError(f"mock payload failed validation for {agent_name}: {exc}") from exc

    meta = LLMMeta(
        provider="mock",
        model="mock",
        agent_name=agent_name,
        tokens=100,
        latency_ms=0,
        raw={},
    )
    _charge(meta.tokens)
    return parsed, meta


def _evidence_ids(pack: Any | None) -> list[str]:
    if pack is None:
        return []
    items = getattr(pack, "items", None)
    if items is None:
        return []
    return [getattr(i, "id", "") for i in items]


def _policy_ids(policy_result: Any | None) -> list[str]:
    if policy_result is None:
        return []
    clauses = getattr(policy_result, "applicable_clauses", None)
    if clauses:
        return list(clauses)
    return []


def _mock_payload(
    agent_name: str,
    schema: type[BaseModel],
    *,
    evidence_pack: Any | None,
    policy_result: Any | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic payload matching the agent's schema fields."""
    fields = schema.model_fields
    eids = _evidence_ids(evidence_pack)
    pids = _policy_ids(policy_result)
    # Fallback IDs when no evidence pack is provided (tests).
    if not eids:
        eids = ["E1", "E2"]
    if not pids:
        pids = ["P-RD-1"]

    # Gather a pool of cite IDs.
    cite_pool = eids + pids

    # Pull some numbers from the evidence pack if available.
    numbers: dict[str, float] = {}
    if evidence_pack is not None:
        for item in getattr(evidence_pack, "items", []):
            data = getattr(item, "data", {})
            for key, val in data.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    numbers[key] = float(val)

    # Amount from policy result if available.
    amount = 0.0
    if policy_result is not None:
        amount = getattr(policy_result, "amount", 0.0)

    payload: dict[str, Any] = {}
    for name, field in fields.items():
        payload[name] = _mock_field(
            name, field, agent_name, cite_pool, eids, pids, numbers, amount,
            evidence_pack, policy_result, context,
        )
    return payload


def _mock_field(
    name: str,
    field: Any,
    agent_name: str,
    cite_pool: list[str],
    eids: list[str],
    pids: list[str],
    numbers: dict[str, float],
    amount: float,
    evidence_pack: Any | None,
    policy_result: Any | None,
    context: dict[str, Any] | None = None,
) -> Any:
    """Generate a deterministic value for one schema field."""
    context = context or {}
    annotation = field.annotation
    origin = getattr(annotation, "__origin__", None)
    # Unwrap Optional / Union.
    if origin is None and hasattr(annotation, "__args__"):
        # Could be Optional[X] = Union[X, None]
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
            origin = getattr(annotation, "__origin__", None)

    # List fields.
    if origin in (list, list):
        return _mock_list_field(
            name, agent_name, cite_pool, eids, pids, numbers, amount,
            evidence_pack, policy_result, context,
        )

    # String fields — dispatch by name heuristics.
    if annotation is str:
        return _mock_str_field(
            name, agent_name, cite_pool, numbers, amount,
            policy_result=policy_result, context=context,
        )

    # Numeric fields.
    if annotation in (int, float):
        return _mock_num_field(name, numbers, amount)

    # Boolean fields.
    if annotation is bool:
        return _mock_bool_field(name, agent_name, context)

    # Literal fields.
    if hasattr(annotation, "__args__") and origin is not list:
        args = getattr(annotation, "__args__", ())
        if args:
            # For fraud risk level, use the assessment from context.
            if name == "risk" and "assessment" in context:
                return getattr(context["assessment"], "risk", "low")
            return args[0]

    # Nested pydantic models.
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _mock_payload(
            "nested",
            annotation,
            evidence_pack=evidence_pack,
            policy_result=policy_result,
            context=context,
        )

    return None


def _mock_list_field(
    name: str,
    agent_name: str,
    cite_pool: list[str],
    eids: list[str],
    pids: list[str],
    numbers: dict[str, float],
    amount: float,
    evidence_pack: Any | None,
    policy_result: Any | None,
    context: dict[str, Any] | None = None,
) -> list[Any]:
    """Build a deterministic list for fields like arguments or signals."""
    context = context or {}
    # Arguments / signals / matches — list of objects with text + cites.
    if name in ("arguments", "signals", "cites", "matches"):
        if name == "arguments":
            return [
                {
                    "text": _mock_str_field(
                        "argument_text", agent_name, cite_pool, numbers, amount,
                        context=context,
                    ),
                    "cites": cite_pool[:2],
                },
                {
                    "text": _mock_str_field(
                        "argument_text_2", agent_name, cite_pool, numbers, amount,
                        context=context,
                    ),
                    "cites": [cite_pool[0]] if cite_pool else ["E1"],
                },
            ]
        if name == "signals":
            # Use the deterministic fraud assessment from context if available.
            assessment = context.get("assessment")
            if assessment is not None:
                return [
                    {"text": s.text, "cites": s.cites}
                    for s in assessment.signals
                ]
            return [
                {
                    "text": "Prior upheld disputes detected in profile history.",
                    "cites": eids[:1] if eids else ["E1"],
                },
            ]
        if name == "cites":
            return cite_pool[:3]
        if name == "matches":
            # Precedent matches from context.
            matches = context.get("matches", [])
            return [
                {
                    "case_id": m.case_id,
                    "category": m.category,
                    "summary": m.summary,
                    "outcome": m.outcome,
                    "amount": m.amount,
                    "similarity": m.similarity,
                    "reason": m.reason,
                    "source": m.source,
                }
                for m in matches
            ]

    # Default: a single-element list of strings citing the first evidence.
    return [cite_pool[0]] if cite_pool else []


def _mock_str_field(
    name: str,
    agent_name: str,
    cite_pool: list[str],
    numbers: dict[str, float],
    amount: float,
    *,
    policy_result: Any | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Deterministic string content that cites real IDs and uses real numbers."""
    context = context or {}
    cite_str = ", ".join(cite_pool[:3])

    if name in ("position", "requested_outcome"):
        if agent_name == "rider_advocate":
            return f"The rider requests a refund based on evidence {cite_str}."
        if agent_name == "driver_advocate":
            return f"The driver requests no refund based on evidence {cite_str}."
        return f"Position based on evidence {cite_str}."

    if name in ("outcome",):
        # Use the policy engine's outcome so the Judge doesn't mismatch.
        if policy_result is not None:
            outcome = getattr(policy_result, "outcome", "")
            if outcome:
                return outcome
        return "refund" if amount > 0 else "no_refund"

    if name in ("reasoning", "message_to_rider", "message_to_driver"):
        amt = amount if amount else numbers.get("charged_total", 0.0)
        parts = [
            f"Based on {cite_str}, the computed amount is S${amt:.2f}. "
            "The policy engine determines the refund."
        ]
        # If bad-faith signals are present, mention them (P-BF-1).
        assessment = context.get("assessment")
        if assessment is not None and getattr(assessment, "risk", "low") == "high":
            parts.append(
                " Bad-faith signals noted: high upheld dispute rate (P-BF-1) "
                "lowers claimant credibility."
            )
        # If precedents were provided, mention consistency.
        matches = context.get("matches")
        if matches:
            parts.append(" Ruling is consistent with retrieved precedents.")
        return " ".join(parts)

    if name in ("text", "argument_text", "argument_text_2", "summary"):
        # For the precedent agent's summary field, use matches context.
        if agent_name == "precedent_agent" and "matches" in context:
            matches = context["matches"]
            if matches:
                return f"Retrieved {len(matches)} similar precedents."
            return "No similar precedents found."
        extra_km = numbers.get("extra_km", 0.0)
        return (
            f"Evidence {cite_str} shows the route deviation of "
            f"{extra_km:.2f} km, supporting the claim."
        )

    if name in ("risk",):
        # This is a str field on FraudOutput.
        assessment = context.get("assessment")
        if assessment is not None:
            return getattr(assessment, "risk", "low")
        return "low"

    if name == "confidence":
        # This is a float field, not a string, handled elsewhere.
        return "0.8"

    if name == "precedent_consistency":
        matches = context.get("matches")
        if matches:
            return "consistent"
        return "no_precedents"

    if name == "reason":
        # Precedent reason.
        matches = context.get("matches", [])
        if matches:
            return matches[0].reason if matches else "keyword overlap"
        return "no match"

    # Default — a neutral sentence that cites the first evidence ID.
    return f"Assessment based on {cite_str}."


def _mock_num_field(name: str, numbers: dict[str, float], amount: float) -> float:
    if name == "amount":
        return round(amount, 2)
    if name == "confidence":
        return 0.8
    # Try to find a matching number from the evidence.
    for key, val in numbers.items():
        if name in key or key in name:
            return round(val, 4)
    return 0.0


def _mock_bool_field(name: str, agent_name: str, context: dict[str, Any] | None = None) -> bool:
    context = context or {}
    if name == "bad_faith_mentioned":
        assessment = context.get("assessment")
        if assessment is not None:
            return getattr(assessment, "risk", "low") == "high"
    return False


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict[str, Any]:
    """Parse possibly-fenced JSON text into a dict."""
    text = text.strip()
    # Strip markdown code fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise LLMError(f"could not parse JSON from LLM response: {text[:200]}")
