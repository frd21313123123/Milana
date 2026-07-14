"""Manifest loading and explicit runtime bindings for Milana skills."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Mapping

from .types import SkillExecutor, SkillSpec

if TYPE_CHECKING:
    from .session import SkillSession


SKILL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*$")
MANIFEST_FIELDS = frozenset(
    {"id", "version", "title", "summary", "parent", "instructions", "children"}
)


class SkillError(Exception):
    """Base error for the skill subsystem."""


class SkillManifestError(SkillError, ValueError):
    """A package manifest or its hierarchy is invalid."""


class SkillRegistrationError(SkillError, ValueError):
    """Python runtime bindings conflict or are malformed."""


ActivationHook = Callable[
    [SkillSpec, "SkillSession"], Any | Awaitable[Any]
]


@dataclass(frozen=True, slots=True)
class SkillBinding:
    """Tools and host code deliberately attached to one validated manifest."""

    tools: tuple[Mapping[str, Any], ...]
    executor: SkillExecutor | None
    on_activate: ActivationHook | None = None


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SkillManifestError(f"Duplicate manifest field {key!r}")
        result[key] = value
    return result


def _required_text(payload: Mapping[str, Any], key: str, source: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillManifestError(f"{source}: {key!r} must be a non-empty string")
    return value.strip()


def load_skill_manifest(manifest_path: str | Path) -> SkillSpec:
    """Load one data-only manifest and its local Markdown instructions."""

    source = Path(manifest_path).resolve()
    if not source.is_file():
        raise SkillManifestError(f"Skill manifest does not exist: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except UnicodeDecodeError as exc:
        raise SkillManifestError(f"{source}: manifest must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise SkillManifestError(f"{source}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise SkillManifestError(f"{source}: manifest root must be a JSON object")
    missing = MANIFEST_FIELDS.difference(payload)
    unknown = set(payload).difference(MANIFEST_FIELDS)
    if missing:
        raise SkillManifestError(
            f"{source}: missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise SkillManifestError(
            f"{source}: unknown fields: {', '.join(sorted(unknown))}"
        )

    skill_id = _required_text(payload, "id", source)
    if not SKILL_ID_PATTERN.fullmatch(skill_id):
        raise SkillManifestError(f"{source}: invalid skill id {skill_id!r}")

    parent = payload["parent"]
    if parent is not None and (
        not isinstance(parent, str) or not SKILL_ID_PATTERN.fullmatch(parent)
    ):
        raise SkillManifestError(f"{source}: parent must be a valid skill id or null")

    raw_children = payload["children"]
    if not isinstance(raw_children, list) or not all(
        isinstance(child, str) and SKILL_ID_PATTERN.fullmatch(child)
        for child in raw_children
    ):
        raise SkillManifestError(f"{source}: children must be a list of skill ids")
    if len(raw_children) != len(set(raw_children)):
        raise SkillManifestError(f"{source}: children contains duplicate ids")

    instruction_name = _required_text(payload, "instructions", source)
    instruction_ref = Path(instruction_name)
    if instruction_ref.is_absolute():
        raise SkillManifestError(f"{source}: instructions path must be relative")
    instruction_path = (source.parent / instruction_ref).resolve()
    try:
        instruction_path.relative_to(source.parent)
    except ValueError as exc:
        raise SkillManifestError(
            f"{source}: instructions path escapes the skill package"
        ) from exc
    if not instruction_path.is_file():
        raise SkillManifestError(
            f"{source}: instructions file does not exist: {instruction_name}"
        )
    try:
        instructions = instruction_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise SkillManifestError(
            f"{instruction_path}: instructions must be UTF-8"
        ) from exc
    if not instructions:
        raise SkillManifestError(f"{instruction_path}: instructions are empty")

    return SkillSpec(
        id=skill_id,
        version=_required_text(payload, "version", source),
        title=_required_text(payload, "title", source),
        summary=_required_text(payload, "summary", source),
        parent=parent,
        instructions=instructions,
        children=tuple(raw_children),
        source_dir=source.parent,
    )


def load_skill_manifests(root: str | Path) -> tuple[SkillSpec, ...]:
    """Discover all manifest packages below ``root`` in deterministic order."""

    directory = Path(root).resolve()
    if not directory.is_dir():
        raise SkillManifestError(f"Skills directory does not exist: {directory}")
    manifests = sorted(directory.rglob("manifest.json"), key=lambda path: path.as_posix())
    if not manifests:
        raise SkillManifestError(f"No skill manifests found below {directory}")
    return tuple(load_skill_manifest(path) for path in manifests)


class SkillRegistry:
    """Immutable hierarchy plus explicit, mutable-at-startup Python bindings."""

    def __init__(self, specs: Iterable[SkillSpec]) -> None:
        by_id: dict[str, SkillSpec] = {}
        for spec in specs:
            if not isinstance(spec, SkillSpec):
                raise TypeError("SkillRegistry accepts only SkillSpec values")
            self._validate_spec(spec)
            if spec.id in by_id:
                raise SkillManifestError(f"Duplicate skill id {spec.id!r}")
            by_id[spec.id] = spec
        if not by_id:
            raise SkillManifestError("A skill registry cannot be empty")
        self._validate_hierarchy(by_id)
        self._specs = MappingProxyType(by_id)
        self._bindings: dict[str, SkillBinding] = {}
        self._tool_owners: dict[str, str] = {}

    @classmethod
    def from_directory(cls, root: str | Path) -> "SkillRegistry":
        return cls(load_skill_manifests(root))

    @staticmethod
    def _validate_spec(spec: SkillSpec) -> None:
        if not SKILL_ID_PATTERN.fullmatch(spec.id):
            raise SkillManifestError(f"Invalid skill id {spec.id!r}")
        for field_name in ("version", "title", "summary", "instructions"):
            value = getattr(spec, field_name)
            if not isinstance(value, str) or not value.strip():
                raise SkillManifestError(
                    f"Skill {spec.id!r} has invalid {field_name!r}"
                )
        if spec.parent is not None and (
            not isinstance(spec.parent, str)
            or not SKILL_ID_PATTERN.fullmatch(spec.parent)
        ):
            raise SkillManifestError(
                f"Skill {spec.id!r} has invalid parent {spec.parent!r}"
            )
        if not isinstance(spec.children, tuple) or not all(
            isinstance(child, str) and SKILL_ID_PATTERN.fullmatch(child)
            for child in spec.children
        ):
            raise SkillManifestError(
                f"Skill {spec.id!r} children must be a tuple of valid ids"
            )
        if len(spec.children) != len(set(spec.children)):
            raise SkillManifestError(f"Skill {spec.id!r} repeats a child id")

    @staticmethod
    def _validate_hierarchy(specs: Mapping[str, SkillSpec]) -> None:
        for spec in specs.values():
            if spec.parent is None:
                if "." in spec.id:
                    raise SkillManifestError(
                        f"Nested skill {spec.id!r} must declare its parent"
                    )
            else:
                expected_parent = spec.id.rpartition(".")[0]
                if not expected_parent or spec.parent != expected_parent:
                    raise SkillManifestError(
                        f"Skill {spec.id!r} must declare direct parent {expected_parent!r}"
                    )
                if spec.parent not in specs:
                    raise SkillManifestError(
                        f"Skill {spec.id!r} references missing parent {spec.parent!r}"
                    )

            for child_id in spec.children:
                child = specs.get(child_id)
                if child is None:
                    raise SkillManifestError(
                        f"Skill {spec.id!r} references missing child {child_id!r}"
                    )
                if child.parent != spec.id:
                    raise SkillManifestError(
                        f"Skill {spec.id!r} lists {child_id!r}, whose parent is "
                        f"{child.parent!r}"
                    )

        for spec in specs.values():
            if spec.parent is not None and spec.id not in specs[spec.parent].children:
                raise SkillManifestError(
                    f"Parent {spec.parent!r} does not list child {spec.id!r}"
                )

        for skill_id in specs:
            seen: set[str] = set()
            current: str | None = skill_id
            while current is not None:
                if current in seen:
                    raise SkillManifestError(f"Cycle detected at skill {current!r}")
                seen.add(current)
                current = specs[current].parent

    @property
    def specs(self) -> Mapping[str, SkillSpec]:
        return self._specs

    @property
    def roots(self) -> tuple[SkillSpec, ...]:
        return tuple(
            sorted(
                (spec for spec in self._specs.values() if spec.parent is None),
                key=lambda spec: spec.id,
            )
        )

    def get(self, skill_id: str) -> SkillSpec:
        try:
            return self._specs[skill_id]
        except KeyError as exc:
            raise SkillManifestError(f"Unknown skill {skill_id!r}") from exc

    def root_catalog(self) -> tuple[dict[str, str], ...]:
        return tuple(spec.catalog_entry() for spec in self.roots)

    def bind(
        self,
        skill_id: str,
        *,
        tools: Iterable[Mapping[str, Any]] = (),
        executor: SkillExecutor | None = None,
        on_activate: ActivationHook | None = None,
    ) -> None:
        """Attach trusted Python code; manifests never name or import executors."""

        self.get(skill_id)
        if skill_id in self._bindings:
            raise SkillRegistrationError(f"Skill {skill_id!r} is already bound")

        normalized: list[Mapping[str, Any]] = []
        names: list[str] = []
        for raw_schema in tools:
            schema = deepcopy(dict(raw_schema))
            name = self._validate_tool_schema(schema, skill_id)
            owner = self._tool_owners.get(name)
            if owner is not None:
                raise SkillRegistrationError(
                    f"Tool {name!r} is already owned by skill {owner!r}"
                )
            if name in names:
                raise SkillRegistrationError(
                    f"Skill {skill_id!r} declares tool {name!r} more than once"
                )
            names.append(name)
            normalized.append(MappingProxyType(schema))
        if normalized and executor is None:
            raise SkillRegistrationError(
                f"Skill {skill_id!r} declares tools but has no executor"
            )
        if executor is not None and not callable(getattr(executor, "execute", None)):
            raise SkillRegistrationError("executor must provide an execute method")
        if on_activate is not None and not callable(on_activate):
            raise SkillRegistrationError("on_activate must be callable or None")

        self._bindings[skill_id] = SkillBinding(
            tools=tuple(normalized), executor=executor, on_activate=on_activate
        )
        self._tool_owners.update((name, skill_id) for name in names)

    @staticmethod
    def _validate_tool_schema(schema: Mapping[str, Any], skill_id: str) -> str:
        if schema.get("type") != "function":
            raise SkillRegistrationError(
                f"Tool for skill {skill_id!r} must have type='function'"
            )
        name = schema.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SkillRegistrationError(
                f"Tool for skill {skill_id!r} must have a non-empty name"
            )
        parameters = schema.get("parameters")
        if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
            raise SkillRegistrationError(
                f"Tool {name!r} must declare an object parameters schema"
            )
        return name

    def binding(self, skill_id: str) -> SkillBinding | None:
        self.get(skill_id)
        return self._bindings.get(skill_id)

    def owner_of_tool(self, tool_name: str) -> str | None:
        return self._tool_owners.get(tool_name)

    def tools_for(self, skill_id: str) -> tuple[dict[str, Any], ...]:
        binding = self.binding(skill_id)
        if binding is None:
            return ()
        return tuple(deepcopy(dict(schema)) for schema in binding.tools)

    def openable_skill_ids(self, active: Iterable[str]) -> tuple[str, ...]:
        """Return only discoverable paths; active IDs remain for idempotent retries."""

        active_set = set(active)
        unknown = active_set.difference(self._specs)
        if unknown:
            raise SkillManifestError(
                f"Active set contains unknown skills: {', '.join(sorted(unknown))}"
            )
        result = set(active_set)
        result.update(spec.id for spec in self.roots)
        for skill_id in active_set:
            result.update(self._specs[skill_id].children)
        return tuple(sorted(result))

    def new_session(
        self,
        *,
        turn_id: str | None = None,
        core_executor: SkillExecutor | None = None,
    ) -> "SkillSession":
        from .session import SkillSession

        return SkillSession(self, turn_id=turn_id, core_executor=core_executor)


def default_skills_root() -> Path:
    return Path(__file__).resolve().parent.parent / "skills"


def load_default_registry() -> SkillRegistry:
    return SkillRegistry.from_directory(default_skills_root())
