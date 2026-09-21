"""Strict asynchronous client for TypeSafe's Jev decision API."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx


LOGGER = logging.getLogger(__name__)
DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Base error for an unusable Jev decision."""


class JevUnavailableError(JevError):
    """Jev cannot be consulted for this turn."""


class JevValidationError(JevError, ValueError):
    """Jev returned a response outside the documented contract."""


class JevLowConfidenceError(JevError):
    """A structurally valid answer is below the configured threshold."""


@dataclass(frozen=True, slots=True)
class JevConfig:
    enabled: bool = False
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_JEV_MODEL
    base_url: str = DEFAULT_JEV_BASE_URL
    confidence_threshold: float = 0.80
    timeout_seconds: float = 3.0
    sticker_timeout_seconds: float = 1.5
    failure_threshold: int = 3
    cooldown_seconds: float = 300.0
    phone_planner: bool = True
    sticker_intent: bool = True
    heartbeat_router: bool = True
    initiative: bool = True
    open_loops: bool = True


@dataclass(frozen=True, slots=True)
class JevChoice:
    choice: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class OpenLoopCandidate:
    id: str
    kind: str
    title: str
    detail: str = ""
    target_ref: str | int | None = None
    actionable: bool = True
    priority: int = 0
    updated_at: str | None = None

    def model_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title[:500],
            "detail": self.detail[:1500],
            "target_ref": self.target_ref,
            "actionable": self.actionable,
            "priority": self.priority,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class OpenLoopSnapshot:
    candidates: tuple[OpenLoopCandidate, ...]

    @property
    def actionable(self) -> tuple[OpenLoopCandidate, ...]:
        return tuple(item for item in self.candidates if item.actionable)

    def model_payload(self) -> list[dict[str, Any]]:
        return [item.model_payload() for item in self.candidates]


@dataclass(frozen=True, slots=True)
class HeartbeatRouteDecision:
    route: str
    target_ref: str | int | None = None
    open_loop_id: str | None = None
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class JevNoul:
    probability: float


@dataclass(frozen=True, slots=True)
class JevResult:
    model: str
    answers: Mapping[str, Mapping[str, Any]]
    input_tokens: int
    output_tokens: int
    elapsed_ms: float

    def choice(
        self,
        name: str,
        *,
        allowed: set[str] | frozenset[str],
        minimum_confidence: float,
    ) -> JevChoice:
        raw = self.answers.get(name)
        if not isinstance(raw, Mapping) or raw.get("type") != "choice":
            raise JevValidationError(f"Jev answer {name!r} is not a Choice")
        selected = raw.get("choice")
        confidence = raw.get("confidence")
        probabilities = raw.get("probabilities")
        if not isinstance(selected, str) or selected not in allowed:
            raise JevValidationError(f"Jev answer {name!r} selected an unknown option")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise JevValidationError(f"Jev answer {name!r} has invalid confidence")
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(allowed):
            raise JevValidationError(f"Jev answer {name!r} has invalid probabilities")
        normalized: dict[str, float] = {}
        for option, value in probabilities.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise JevValidationError(
                    f"Jev answer {name!r} has an invalid probability"
                )
            normalized[str(option)] = float(value)
        if abs(sum(normalized.values()) - 1.0) > 0.02:
            raise JevValidationError(f"Jev answer {name!r} probabilities do not sum to 1")
        decision = JevChoice(selected, float(confidence), normalized)
        if decision.confidence < minimum_confidence:
            raise JevLowConfidenceError(
                f"Jev answer {name!r} confidence {decision.confidence:.3f} "
                f"is below {minimum_confidence:.3f}"
            )
        return decision

    def noul(self, name: str) -> JevNoul:
        raw = self.answers.get(name)
        if not isinstance(raw, Mapping) or raw.get("type") != "noul":
            raise JevValidationError(f"Jev answer {name!r} is not a Noul")
        value = raw.get("noul")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 1
        ):
            raise JevValidationError(f"Jev answer {name!r} has invalid probability")
        return JevNoul(float(value))


class JevDecisionClient:
    """One-shot Jev requests with a small in-process circuit breaker."""

    def __init__(
        self,
        config: JevConfig,
        *,
        client: httpx.AsyncClient | None = None,
        monotonic: Any = time.monotonic,
    ) -> None:
        if not config.enabled or not config.api_key:
            raise ValueError("JevDecisionClient requires an enabled config and API key")
        self.config = config
        self._authorization = f"Bearer {config.api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._owns_client = client is None
        self._monotonic = monotonic
        self._failures = 0
        self._open_until = 0.0
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    @property
    def circuit_open(self) -> bool:
        return self._open_until > self._monotonic()

    async def evaluate(
        self,
        *,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        scenario: str,
        timeout_seconds: float | None = None,
    ) -> JevResult:
        if not questions:
            raise ValueError("Jev questions cannot be empty")
        async with self._lock:
            now = self._monotonic()
            if self._open_until > now:
                raise JevUnavailableError("Jev circuit breaker is open")
            if self._open_until and self._probe_in_flight:
                raise JevUnavailableError("Jev recovery probe is already running")
            if self._open_until:
                self._probe_in_flight = True

        started = self._monotonic()
        try:
            response = await self._client.post(
                "/v1/systemone",
                headers={
                    "Authorization": self._authorization,
                    "Content-Type": "application/json",
                },
                json={
                    "state": state,
                    "model": self.config.model,
                    "questions": dict(questions),
                },
                timeout=timeout_seconds or self.config.timeout_seconds,
            )
            if response.status_code != 200:
                raise JevUnavailableError(f"Jev returned HTTP {response.status_code}")
            try:
                payload = response.json()
            except ValueError as exc:
                raise JevValidationError("Jev returned invalid JSON") from exc
            result = self._validate_result(payload, (self._monotonic() - started) * 1000)
        except asyncio.CancelledError:
            async with self._lock:
                self._probe_in_flight = False
            raise
        except Exception as exc:
            await self._record_failure(
                scenario,
                exc,
                elapsed_ms=(self._monotonic() - started) * 1000,
            )
            if isinstance(exc, JevError):
                raise
            raise JevUnavailableError(f"{type(exc).__name__}: {exc}") from exc

        async with self._lock:
            self._failures = 0
            self._open_until = 0.0
            self._probe_in_flight = False
        LOGGER.info(
            "Jev scenario=%s model=%s elapsed_ms=%.1f input_tokens=%d output_tokens=%d",
            scenario,
            result.model,
            result.elapsed_ms,
            result.input_tokens,
            result.output_tokens,
        )
        return result

    async def _record_failure(
        self,
        scenario: str,
        exc: Exception,
        *,
        elapsed_ms: float | None = None,
    ) -> None:
        async with self._lock:
            self._probe_in_flight = False
            self._failures += 1
            if self._failures >= self.config.failure_threshold:
                self._open_until = self._monotonic() + self.config.cooldown_seconds
        LOGGER.warning(
            "Jev fallback scenario=%s reason=%s model=%s elapsed_ms=%s "
            "input_tokens=%s output_tokens=%s failures=%d",
            scenario,
            type(exc).__name__,
            self.config.model,
            elapsed_ms,
            None,
            None,
            self._failures,
        )

    async def record_rejection(self, scenario: str, exc: Exception) -> None:
        """Count a caller-side contract rejection without retrying the request."""

        await self._record_failure(scenario, exc)

    @staticmethod
    def _validate_result(payload: Any, elapsed_ms: float) -> JevResult:
        if not isinstance(payload, Mapping):
            raise JevValidationError("Jev response must be an object")
        model = payload.get("model")
        answers = payload.get("answers")
        usage = payload.get("usage")
        if not isinstance(model, str) or not model.strip():
            raise JevValidationError("Jev response has no model")
        if not isinstance(answers, Mapping):
            raise JevValidationError("Jev response has no answers")
        if not isinstance(usage, Mapping):
            raise JevValidationError("Jev response has no usage")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or input_tokens < 0
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 0
        ):
            raise JevValidationError("Jev response has invalid usage")
        return JevResult(
            model=model.strip(),
            answers={str(key): value for key, value in answers.items()},
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed_ms,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
