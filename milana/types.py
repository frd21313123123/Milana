"""Provider-neutral types shared by Milana's agent and skill hosts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .session import SkillSession


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """Validated metadata and lazily disclosed instructions for one skill."""

    id: str
    version: str
    title: str
    summary: str
    parent: str | None
    instructions: str
    children: tuple[str, ...] = ()
    source_dir: Path | None = field(default=None, repr=False, compare=False)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def catalog_entry(self) -> dict[str, str]:
        """Return safe discovery metadata without leaking skill instructions."""

        return {"id": self.id, "title": self.title, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model tool call independent of OpenAI, Gemini, or AGY response classes."""

    name: str
    arguments_json: str = "{}"
    call_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolCall.name must be a non-empty string")
        if not isinstance(self.arguments_json, str):
            raise TypeError("ToolCall.arguments_json must be a JSON string")
        if self.call_id is not None and not isinstance(self.call_id, str):
            raise TypeError("ToolCall.call_id must be a string or None")

    @classmethod
    def from_arguments(
        cls,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> "ToolCall":
        return cls(
            name=name,
            arguments_json=json.dumps(
                dict(arguments or {}), ensure_ascii=False, separators=(",", ":")
            ),
            call_id=call_id,
        )

    def parse_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON arguments for tool {self.name!r}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Arguments for tool {self.name!r} must be a JSON object")
        return value

    @property
    def arguments(self) -> dict[str, Any]:
        """Parsed arguments; primarily a convenience for skill executors."""

        return self.parse_arguments()


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized success or failure returned to the model after a tool call."""

    name: str
    output: Any = None
    call_id: str | None = None
    ok: bool = True
    error: str | None = None
    arguments_json: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolResult.name must be a non-empty string")
        if self.call_id is not None and not isinstance(self.call_id, str):
            raise TypeError("ToolResult.call_id must be a string or None")
        if self.arguments_json is not None and not isinstance(
            self.arguments_json, str
        ):
            raise TypeError("ToolResult.arguments_json must be a string or None")
        if self.ok and self.error is not None:
            raise ValueError("A successful ToolResult cannot contain an error")
        if not self.ok and (not isinstance(self.error, str) or not self.error):
            raise ValueError("A failed ToolResult must contain a non-empty error")

    @classmethod
    def success(cls, call: ToolCall, output: Any = None) -> "ToolResult":
        return cls(
            name=call.name,
            output=output,
            call_id=call.call_id,
            arguments_json=call.arguments_json,
        )

    @classmethod
    def failure(cls, call: ToolCall, error: str) -> "ToolResult":
        return cls(
            name=call.name,
            call_id=call.call_id,
            ok=False,
            error=error,
            arguments_json=call.arguments_json,
        )

    def model_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["output"] = self.output
        else:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True, slots=True)
class ModelStep:
    """Exactly one provider-neutral model step: calls or a final payload."""

    tool_calls: tuple[ToolCall, ...] = ()
    final_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        has_calls = bool(self.tool_calls)
        has_final = self.final_payload is not None
        if has_calls == has_final:
            raise ValueError(
                "ModelStep must contain either tool_calls or final_payload, but not both"
            )
        if not all(isinstance(call, ToolCall) for call in self.tool_calls):
            raise TypeError("ModelStep.tool_calls must contain only ToolCall values")
        if self.final_payload is not None and not isinstance(
            self.final_payload, Mapping
        ):
            raise TypeError("ModelStep.final_payload must be a mapping or None")

    @classmethod
    def calls(cls, *calls: ToolCall) -> "ModelStep":
        return cls(tool_calls=tuple(calls))

    @classmethod
    def final(cls, payload: Mapping[str, Any]) -> "ModelStep":
        return cls(final_payload=dict(payload))

    @property
    def is_final(self) -> bool:
        return self.final_payload is not None


@runtime_checkable
class SkillExecutor(Protocol):
    """Explicit Python binding between a manifest and its side-effecting host."""

    async def execute(
        self,
        call: ToolCall,
        *,
        session: "SkillSession",
    ) -> ToolResult | Any:
        """Execute an already-authorized tool call for ``session``."""
