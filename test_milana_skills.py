from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from milana import (
    ModelStep,
    SkillActivationError,
    SkillManifestError,
    SkillNotActiveError,
    SkillRegistrationError,
    SkillRegistry,
    SkillSpec,
    ToolCall,
    ToolResult,
    bind_telegram_skill_tree,
    load_default_registry,
    load_skill_manifest,
)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def execute(self, call: ToolCall, *, session: Any) -> ToolResult:
        self.calls.append(call)
        return ToolResult.success(
            call,
            {"arguments": call.arguments, "active": session.active_skill_ids},
        )


def tool_names(session: Any) -> list[str]:
    return [tool["name"] for tool in session.tools]


class DefaultSkillManifestTests(unittest.TestCase):
    def test_loads_hierarchy_and_root_catalog_hides_child(self) -> None:
        registry = load_default_registry()

        self.assertEqual(set(registry.specs), {"telegram", "telegram.stickers"})
        self.assertEqual(registry.get("telegram.stickers").parent, "telegram")
        self.assertIn("open_sticker_picker", registry.get("telegram.stickers").instructions)
        self.assertEqual(
            [item["id"] for item in registry.root_catalog()], ["telegram"]
        )
        self.assertNotIn("stickers", json.dumps(registry.root_catalog()))

    def test_rejects_instruction_path_outside_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "skill"
            package.mkdir()
            (root / "outside.md").write_text("secret", encoding="utf-8")
            manifest = {
                "id": "unsafe",
                "version": "1.0.0",
                "title": "Unsafe",
                "summary": "Unsafe test",
                "parent": None,
                "instructions": "../outside.md",
                "children": [],
            }
            (package / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            with self.assertRaisesRegex(SkillManifestError, "escapes"):
                load_skill_manifest(package / "manifest.json")

    def test_rejects_inconsistent_bidirectional_hierarchy(self) -> None:
        parent = SkillSpec(
            id="parent",
            version="1",
            title="Parent",
            summary="Parent",
            parent=None,
            instructions="parent instructions",
            children=(),
        )
        child = SkillSpec(
            id="parent.child",
            version="1",
            title="Child",
            summary="Child",
            parent="parent",
            instructions="child instructions",
            children=(),
        )

        with self.assertRaisesRegex(SkillManifestError, "does not list child"):
            SkillRegistry((parent, child))

    def test_rejects_duplicate_tool_ownership(self) -> None:
        registry = load_default_registry()
        executor = FakeExecutor()
        schema = {
            "type": "function",
            "name": "same_tool",
            "parameters": {"type": "object", "properties": {}},
        }
        registry.bind("telegram", tools=[schema], executor=executor)

        with self.assertRaisesRegex(SkillRegistrationError, "already owned"):
            registry.bind("telegram.stickers", tools=[schema], executor=executor)


class SkillSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.telegram = FakeExecutor()
        self.stickers = FakeExecutor()
        self.registry = load_default_registry()
        bind_telegram_skill_tree(
            self.registry,
            telegram_executor=self.telegram,
            sticker_executor=self.stickers,
        )

    async def test_open_skill_enum_and_tools_expand_in_parent_order(self) -> None:
        session = self.registry.new_session(turn_id="turn-1")

        self.assertEqual(session.discoverable_skill_ids(), ("telegram",))
        self.assertEqual(
            session.open_skill_tool()["parameters"]["properties"]["skill_id"]["enum"],
            ["telegram"],
        )
        self.assertEqual(
            tool_names(session),
            ["open_skill", "write_diary", "inspect_schedule", "schedule_wakeup"],
        )

        opened = await session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )
        self.assertTrue(opened.ok)
        self.assertEqual(opened.output["skill"]["id"], "telegram")
        self.assertEqual(
            [child["id"] for child in opened.output["children"]],
            ["telegram.stickers"],
        )
        self.assertEqual(
            session.discoverable_skill_ids(),
            ("telegram", "telegram.stickers"),
        )
        self.assertIn("schedule_message", tool_names(session))
        self.assertNotIn("open_sticker_picker", tool_names(session))

        await session.execute_tool(
            ToolCall.from_arguments(
                "open_skill", {"skill_id": "telegram.stickers"}
            )
        )
        self.assertIn("open_sticker_picker", tool_names(session))
        self.assertIn("send_sticker", tool_names(session))
        self.assertIn("schedule_sticker", tool_names(session))

    async def test_strict_parent_and_tool_gating_has_no_side_effect(self) -> None:
        session = self.registry.new_session()

        with self.assertRaisesRegex(SkillActivationError, "parent 'telegram'"):
            await session.execute_tool(
                ToolCall.from_arguments(
                    "open_skill", {"skill_id": "telegram.stickers"}
                )
            )
        with self.assertRaisesRegex(SkillNotActiveError, "telegram.stickers"):
            await session.execute_tool(
                ToolCall.from_arguments("send_sticker", {"sticker_id": "invented"})
            )
        self.assertEqual(self.stickers.calls, [])

        await session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )
        with self.assertRaises(SkillNotActiveError):
            await session.execute_tool(
                ToolCall.from_arguments("open_sticker_picker", {"pack_id": None})
            )
        self.assertEqual(self.stickers.calls, [])

    async def test_activation_is_idempotent_and_hook_runs_once(self) -> None:
        registry = load_default_registry()
        activation_count = 0

        async def activate(spec: SkillSpec, session: Any) -> dict[str, Any]:
            nonlocal activation_count
            activation_count += 1
            return {"notice_id": 42, "skill": spec.id, "turn": session.turn_id}

        registry.bind(
            "telegram",
            tools=[],
            on_activate=activate,
        )
        session = registry.new_session(turn_id="turn-id")
        call = ToolCall.from_arguments(
            "open_skill", {"skill_id": "telegram"}, call_id="one"
        )

        first = await session.execute_tool(call)
        second = await session.execute_tool(
            ToolCall.from_arguments(
                "open_skill", {"skill_id": "telegram"}, call_id="two"
            )
        )

        self.assertEqual(activation_count, 1)
        self.assertEqual(first.output, second.output)
        self.assertEqual(first.output["context"]["turn"], "turn-id")
        self.assertEqual(session.active_skill_ids, ("telegram",))

    async def test_sessions_are_isolated_per_turn(self) -> None:
        first = self.registry.new_session(turn_id="first")
        second = self.registry.new_session(turn_id="second")

        await first.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )
        await first.execute_tool(
            ToolCall.from_arguments(
                "open_skill", {"skill_id": "telegram.stickers"}
            )
        )

        self.assertEqual(
            first.active_skill_ids, ("telegram", "telegram.stickers")
        )
        self.assertEqual(second.active_skill_ids, ())
        self.assertEqual(second.discoverable_skill_ids(), ("telegram",))
        self.assertNotIn("send_sticker", tool_names(second))

    async def test_authorized_tool_is_dispatched_with_parsed_arguments(self) -> None:
        session = self.registry.new_session()
        await session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )
        await session.execute_tool(
            ToolCall.from_arguments(
                "open_skill", {"skill_id": "telegram.stickers"}
            )
        )
        call = ToolCall.from_arguments(
            "open_sticker_picker", {"pack_id": None}, call_id="picker"
        )

        result = await session.execute_tool(call)

        self.assertEqual(self.stickers.calls, [call])
        self.assertEqual(result.call_id, "picker")
        self.assertEqual(result.output["arguments"], {"pack_id": None})


class ProviderNeutralTypeTests(unittest.TestCase):
    def test_tool_call_requires_json_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            ToolCall("bad", "[]").parse_arguments()

    def test_model_step_is_exclusive(self) -> None:
        call = ToolCall.from_arguments("tool")
        self.assertFalse(ModelStep.calls(call).is_final)
        self.assertTrue(ModelStep.final({"messages": []}).is_final)

        with self.assertRaisesRegex(ValueError, "either"):
            ModelStep()
        with self.assertRaisesRegex(ValueError, "either"):
            ModelStep(tool_calls=(call,), final_payload={"messages": []})


if __name__ == "__main__":
    unittest.main()
