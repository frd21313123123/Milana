import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from milana import (
    MilanaAgent,
    SkillActivationRequired,
    ToolResult,
    TurnTrigger,
    bind_telegram_skill_tree,
    empty_turn_payload,
    load_default_registry,
)


class _Responses:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("unexpected model call")
        return self._responses.pop(0)


class _ModelClient:
    def __init__(self, responses):
        self.responses = _Responses(responses)


class _Executor:
    def __init__(self):
        self.calls = []

    async def execute(self, call, *, session):
        self.calls.append((session.turn_id, call.name, call.arguments))
        return ToolResult.success(call, {"accepted": True})


def _function_call(name, arguments, call_id):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
                call_id=call_id,
            )
        ],
        output_text="",
    )


def _final(payload):
    return SimpleNamespace(output=[], output_text=json.dumps(payload, ensure_ascii=False))


def _tool_names(request):
    return {tool["name"] for tool in request["tools"]}


class MilanaAgentTests(unittest.IsolatedAsyncioTestCase):
    def _agent(
        self,
        responses,
        *,
        on_activate=None,
        tool_result_content=None,
    ):
        registry = load_default_registry()
        core = _Executor()
        telegram = _Executor()
        stickers = _Executor()
        bind_telegram_skill_tree(
            registry,
            telegram_executor=telegram,
            sticker_executor=stickers,
            telegram_on_activate=on_activate,
        )
        model = _ModelClient(responses)
        agent = MilanaAgent(
            model,
            model="fake-model",
            persona="Милана остаётся собой.",
            registry=registry,
            core_executor=core,
            tool_result_content=tool_result_content,
        )
        return agent, model, core, telegram, stickers

    def test_provider_step_cannot_mix_tool_calls_with_final_payload(self):
        response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="inspect_schedule",
                    arguments="{}",
                    call_id="mixed",
                )
            ],
            output_text=json.dumps(empty_turn_payload()),
        )

        with self.assertRaisesRegex(ValueError, "tool calls and a final payload"):
            MilanaAgent._normalize_step(response)

    async def test_flattened_provider_state_is_restored_before_validation(self):
        flattened = {
            "mood_label": "задумчивое",
            "valence": 5,
            "arousal": 45,
            "social": 50,
            "rest": 50,
            "novelty": 50,
            "achievement": 50,
            "current_intention": "ответить после пробуждения",
        }
        agent, *_ = self._agent([_final(flattened)])

        result = await agent.run_turn(
            TurnTrigger(kind="heartbeat", occurred_at=datetime.now(timezone.utc))
        )

        self.assertEqual(result.payload["state_update"]["mood_label"], "задумчивое")
        self.assertEqual(result.payload["entity_updates"], [])
        self.assertEqual(result.payload["life_events"], [])
        self.assertEqual(result.payload["goal_updates"], [])
        self.assertEqual(result.payload["relationship_updates"], [])

    async def test_invalid_telegram_final_is_corrected_inside_the_same_turn(self):
        valid = empty_turn_payload(telegram=True)
        valid["telegram"] = {
            "target_token": "token-for-turn",
            "messages": ["второй ответ"],
            "reaction": None,
            "blacklist_sender": False,
        }
        agent, model, *_ = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(empty_turn_payload()),
                _final(valid),
            ]
        )

        result = await agent.run_turn(
            TurnTrigger(
                kind="telegram_notice",
                occurred_at=datetime.now(timezone.utc),
                metadata={"notice_ids": ["tg:77:10"]},
            )
        )

        self.assertEqual(result.payload["telegram"]["messages"], ["второй ответ"])
        self.assertEqual(len(model.responses.requests), 3)
        self.assertEqual(model.responses.requests[2]["tools"], [])
        correction = model.responses.requests[2]["input"][-1]["content"]
        self.assertIn("Обязательно верни ветку telegram", correction)

    async def test_heartbeat_starts_without_telegram_schema_or_tools(self):
        agent, model, *_ = self._agent([_final(empty_turn_payload())])
        result = await agent.run_turn(
            TurnTrigger(kind="heartbeat", occurred_at=datetime.now(timezone.utc))
        )

        self.assertEqual(result.active_skills, ())
        self.assertEqual(result.validated_changes["entity_updates"], ())
        request = model.responses.requests[0]
        self.assertNotIn("telegram", request["text"]["format"]["schema"]["properties"])
        self.assertEqual(
            _tool_names(request),
            {"open_skill", "write_diary", "inspect_schedule", "schedule_wakeup"},
        )
        enum = request["tools"][0]["parameters"]["properties"]["skill_id"]["enum"]
        self.assertEqual(enum, ["telegram"])
        self.assertNotIn("stickers", request["instructions"].lower())

    async def test_telegram_notice_must_open_skill_before_context_and_reply(self):
        activations = []

        async def activate(spec, session):
            activations.append((spec.id, session.turn_id))
            return {
                "target_token": "turn-only-token",
                "messages": [{"id": 7, "text": "Привет"}],
            }

        agent, model, *_ = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "open-1"),
                _final(empty_turn_payload(telegram=True)),
            ],
            on_activate=activate,
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            source_skill="telegram",
            occurred_at=datetime.now(timezone.utc),
            metadata={"chat_id": 10, "message_id": 7, "media_type": "text"},
        )
        result = await agent.run_turn(trigger)

        self.assertEqual(result.active_skills, ("telegram",))
        self.assertEqual(activations, [("telegram", trigger.id)])
        first, second = model.responses.requests
        self.assertNotIn("schedule_message", _tool_names(first))
        self.assertIn("schedule_message", _tool_names(second))
        self.assertNotIn(
            "telegram",
            first["text"]["format"]["schema"]["properties"],
        )
        self.assertIn(
            "telegram",
            second["text"]["format"]["schema"]["properties"],
        )
        second_open = next(tool for tool in second["tools"] if tool["name"] == "open_skill")
        self.assertEqual(
            second_open["parameters"]["properties"]["skill_id"]["enum"],
            ["telegram", "telegram.stickers"],
        )

    async def test_notice_cannot_finish_without_telegram_activation(self):
        agent, *_ = self._agent(
            [_final(empty_turn_payload()), _final(empty_turn_payload())]
        )
        with self.assertRaises(SkillActivationRequired):
            await agent.run_turn(
                TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=datetime.now(timezone.utc),
                )
            )

    async def test_sticker_tool_appears_only_after_parent_then_child(self):
        agent, model, _, _, stickers = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "open-tg"),
                _function_call(
                    "open_skill",
                    {"skill_id": "telegram.stickers"},
                    "open-stickers",
                ),
                _function_call("open_sticker_picker", {"pack_id": None}, "picker"),
                _final(empty_turn_payload(telegram=True)),
            ]
        )
        result = await agent.run_turn(
            TurnTrigger(kind="heartbeat", occurred_at=datetime.now(timezone.utc))
        )

        self.assertEqual(result.active_skills, ("telegram", "telegram.stickers"))
        self.assertNotIn("open_sticker_picker", _tool_names(model.responses.requests[0]))
        self.assertNotIn("open_sticker_picker", _tool_names(model.responses.requests[1]))
        self.assertIn("open_sticker_picker", _tool_names(model.responses.requests[2]))
        self.assertEqual(stickers.calls[0][1:], ("open_sticker_picker", {"pack_id": None}))

    async def test_each_turn_gets_a_fresh_skill_session(self):
        agent, model, *_ = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "one"),
                _final(empty_turn_payload(telegram=True)),
                _final(empty_turn_payload()),
            ]
        )
        now = datetime.now(timezone.utc)
        await agent.run_turn(TurnTrigger(kind="heartbeat", occurred_at=now))
        second = await agent.run_turn(TurnTrigger(kind="heartbeat", occurred_at=now))

        self.assertEqual(second.active_skills, ())
        third_request = model.responses.requests[2]
        self.assertNotIn("schedule_message", _tool_names(third_request))
        self.assertNotIn(
            "telegram", third_request["text"]["format"]["schema"]["properties"]
        )

    async def test_media_from_opened_skill_is_added_to_the_next_model_step(self):
        async def activate(_spec, _session):
            return {"messages": [{"media_path": "runtime/photo.png"}]}

        seen_results = []

        def media_content(result):
            seen_results.append(result.name)
            return [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,eA==",
                }
            ]

        agent, model, *_ = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(empty_turn_payload(telegram=True)),
            ],
            on_activate=activate,
            tool_result_content=media_content,
        )
        await agent.run_turn(
            TurnTrigger(kind="heartbeat", occurred_at=datetime.now(timezone.utc))
        )

        self.assertEqual(seen_results, ["open_skill"])
        media_messages = [
            item
            for item in model.responses.requests[1]["input"]
            if isinstance(item, dict) and isinstance(item.get("content"), list)
        ]
        self.assertTrue(
            any(
                part.get("type") == "input_image"
                for item in media_messages
                for part in item["content"]
                if isinstance(part, dict)
            )
        )

    async def test_runtime_validation_rejects_invalid_fallback_payload(self):
        invalid = empty_turn_payload(telegram=True)
        invalid["telegram"] = {
            "target_token": "turn-token",
            "messages": ["ответ"],
            "reaction": "не emoji из enum",
            "blacklist_sender": False,
        }
        agent, *_ = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(invalid),
                _final(invalid),
                _final(invalid),
            ]
        )

        with self.assertRaisesRegex(ValueError, "reaction"):
            await agent.run_turn(
                TurnTrigger(kind="heartbeat", occurred_at=datetime.now(timezone.utc))
            )


if __name__ == "__main__":
    unittest.main()
