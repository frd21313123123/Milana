"""Per-turn activation and authorization of hierarchical Milana skills."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any, Mapping

from .registry import SkillError, SkillManifestError, SkillRegistry
from .types import SkillExecutor, SkillSpec, ToolCall, ToolResult


MAX_WAKEUP_DELAY_SECONDS = 30 * 24 * 60 * 60

WRITE_DIARY_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "write_diary",
    "description": "Записать важное личное событие, мысль или чувство в дневник Миланы.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"entry": {"type": "string", "minLength": 1}},
        "required": ["entry"],
        "additionalProperties": False,
    },
}

INSPECT_SCHEDULE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "inspect_schedule",
    "description": "Посмотреть текущее занятие и ближайшие переходы расписания Миланы.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}

SCHEDULE_WAKEUP_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "schedule_wakeup",
    "description": (
        "Разбудить Милану для будущего внутреннего хода через указанное число секунд; "
        "горизонт не больше 30 дней."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_WAKEUP_DELAY_SECONDS,
            },
            "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        },
        "required": ["delay_seconds", "reason"],
        "additionalProperties": False,
    },
}

CORE_TOOLS = (WRITE_DIARY_TOOL, INSPECT_SCHEDULE_TOOL, SCHEDULE_WAKEUP_TOOL)
CORE_TOOL_NAMES = frozenset(tool["name"] for tool in CORE_TOOLS)


class SkillActivationError(SkillError, ValueError):
    """A skill cannot be activated from the current session state."""


class SkillNotActiveError(SkillError, PermissionError):
    """A known tool was called before its owning skill was activated."""


class UnknownToolError(SkillError, LookupError):
    """A model requested a tool that is not registered."""


class SkillSession:
    """Ephemeral activation state owned by exactly one Milana model turn."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        turn_id: str | None = None,
        core_executor: SkillExecutor | None = None,
    ) -> None:
        if not isinstance(registry, SkillRegistry):
            raise TypeError("registry must be a SkillRegistry")
        self.registry = registry
        self.turn_id = turn_id
        self.core_executor = core_executor
        self._active: set[str] = set()
        self._activation_payloads: dict[str, dict[str, Any]] = {}

    @property
    def active_skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    def is_active(self, skill_id: str) -> bool:
        return skill_id in self._active

    def discoverable_skill_ids(self) -> tuple[str, ...]:
        return self.registry.openable_skill_ids(self._active)

    def open_skill_tool(self) -> dict[str, Any]:
        """Build a fresh schema whose enum reflects this turn's activation state."""

        return {
            "type": "function",
            "name": "open_skill",
            "description": (
                "Активировать внешний навык на текущий ход. Сначала открывай родительский "
                "навык; повторное открытие безопасно и ничего не дублирует."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "enum": list(self.discoverable_skill_ids()),
                    }
                },
                "required": ["skill_id"],
                "additionalProperties": False,
            },
        }

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        result = [self.open_skill_tool(), *(deepcopy(tool) for tool in CORE_TOOLS)]
        for skill_id in self.active_skill_ids:
            result.extend(self.registry.tools_for(skill_id))
        return tuple(result)

    def available_tools(self) -> tuple[dict[str, Any], ...]:
        """Method form of :attr:`tools` for request-builder integrations."""

        return self.tools

    def _validate_activation(self, skill_id: str) -> SkillSpec:
        try:
            spec = self.registry.get(skill_id)
        except SkillManifestError as exc:
            raise SkillActivationError(str(exc)) from exc
        if spec.parent is not None and spec.parent not in self._active:
            raise SkillActivationError(
                f"Cannot open skill {skill_id!r}: parent {spec.parent!r} is not active"
            )
        if skill_id not in self.discoverable_skill_ids():
            raise SkillActivationError(
                f"Skill {skill_id!r} is not discoverable in the current session"
            )
        return spec

    def _activation_payload(
        self,
        spec: SkillSpec,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        children = [
            self.registry.get(child_id).catalog_entry() for child_id in spec.children
        ]
        payload: dict[str, Any] = {
            "skill": spec.catalog_entry(),
            "instructions": spec.instructions,
            "children": children,
        }
        if context is not None:
            payload["context"] = context
        return payload

    def open_skill(self, skill_id: str) -> dict[str, Any]:
        """Synchronously activate metadata; useful when no host activation hook exists."""

        if skill_id in self._active:
            return deepcopy(self._activation_payloads[skill_id])
        spec = self._validate_activation(skill_id)
        payload = self._activation_payload(spec)
        self._active.add(skill_id)
        self._activation_payloads[skill_id] = payload
        return deepcopy(payload)

    activate = open_skill

    async def _execute_open_skill(self, call: ToolCall) -> ToolResult:
        arguments = call.parse_arguments()
        if set(arguments) != {"skill_id"} or not isinstance(
            arguments.get("skill_id"), str
        ):
            raise SkillActivationError(
                "open_skill requires exactly one string argument: skill_id"
            )
        skill_id = arguments["skill_id"]
        if skill_id in self._active:
            return ToolResult.success(
                call, deepcopy(self._activation_payloads[skill_id])
            )

        spec = self._validate_activation(skill_id)
        binding = self.registry.binding(skill_id)
        context: Any = None
        if binding is not None and binding.on_activate is not None:
            context = binding.on_activate(spec, self)
            if inspect.isawaitable(context):
                context = await context

        payload = self._activation_payload(spec, context=context)
        self._active.add(skill_id)
        self._activation_payloads[skill_id] = payload
        return ToolResult.success(call, deepcopy(payload))

    async def execute_tool(self, call: ToolCall) -> ToolResult:
        """Authorize first, then dispatch; denied calls never reach a host executor."""

        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall")
        if call.name == "open_skill":
            return await self._execute_open_skill(call)
        if call.name in CORE_TOOL_NAMES:
            if self.core_executor is None:
                raise UnknownToolError(
                    f"Core tool {call.name!r} has no configured executor"
                )
            return await self._dispatch(self.core_executor, call)

        owner = self.registry.owner_of_tool(call.name)
        if owner is None:
            raise UnknownToolError(f"Unknown tool {call.name!r}")
        if owner not in self._active:
            raise SkillNotActiveError(
                f"Tool {call.name!r} requires active skill {owner!r}"
            )
        binding = self.registry.binding(owner)
        if binding is None or binding.executor is None:
            raise UnknownToolError(
                f"Skill {owner!r} has no executor for tool {call.name!r}"
            )
        return await self._dispatch(binding.executor, call)

    execute = execute_tool

    async def _dispatch(
        self, executor: SkillExecutor, call: ToolCall
    ) -> ToolResult:
        value = executor.execute(call, session=self)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, ToolResult):
            if value.name != call.name:
                raise ValueError(
                    f"Executor returned result for {value.name!r}, expected {call.name!r}"
                )
            return value
        return ToolResult.success(call, value)

    def catalog_prompt(self) -> str:
        """Permanent-prompt fragment that exposes root skills only."""

        lines = [
            "Для внешних возможностей используй open_skill. Сначала открой корневой "
            "навык; дочерние навыки станут видны только после открытия родителя.",
            "Доступные корневые навыки:",
        ]
        lines.extend(
            f"- {item['id']}: {item['summary']}"
            for item in self.registry.root_catalog()
        )
        return "\n".join(lines)
