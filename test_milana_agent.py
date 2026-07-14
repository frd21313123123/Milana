import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from milana import (
    MilanaAgent,
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


def _telegram_final(
    message="ответ",
    *,
    token="turn-only-token",
):
    return {
        "telegram": {
            "target_token": token,
            "messages": [message],
            "reaction": None,
            "blacklist_sender": False,
        }
    }


def _tool_names(request):
    return {tool["name"] for tool in request["tools"]}


def _production_notice_metadata(*, chat_id=10, message_id=7):
    notice_id = f"tg:{chat_id}:{message_id}"
    notice = {
        "source": "telegram",
        "notice_id": notice_id,
        "chat_id": chat_id,
        "message_id": message_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender": {"id": 88, "display_name": "Лера"},
        "media_type": "text",
    }
    return {
        "chat_id": chat_id,
        "notice_ids": [notice_id],
        "notices": [notice],
    }


class MilanaAgentTests(unittest.IsolatedAsyncioTestCase):
    def _agent(
        self,
        responses,
        *,
        on_activate=None,
        tool_result_content=None,
        telegram_fast_enabled=False,
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
            telegram_fast_enabled=telegram_fast_enabled,
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

    async def test_fast_invalid_telegram_final_fails_without_correction_round(self):
        agent, model, *_ = self._agent(
            [_final({})],
            telegram_fast_enabled=True,
        )

        with self.assertRaisesRegex(ValueError, "Final payload fields"):
            await agent.run_turn(
                TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=datetime.now(timezone.utc),
                    metadata={"notice_ids": ["tg:77:10"]},
                )
            )
        self.assertEqual(len(model.responses.requests), 1)

    async def test_heartbeat_starts_without_telegram_schema_or_tools(self):
        agent, model, *_ = self._agent([_final(empty_turn_payload())])
        result = await agent.run_turn(
            TurnTrigger(kind="heartbeat", occurred_at=datetime.now(timezone.utc))
        )

        self.assertEqual(result.active_skills, ())
        self.assertEqual(result.validated_changes["entity_updates"], ())
        request = model.responses.requests[0]
        self.assertEqual(request["metadata"], {"agy_priority": "background"})
        self.assertNotIn("telegram", request["text"]["format"]["schema"]["properties"])
        self.assertEqual(
            _tool_names(request),
            {"open_skill", "write_diary", "inspect_schedule", "schedule_wakeup"},
        )
        enum = request["tools"][0]["parameters"]["properties"]["skill_id"]["enum"]
        self.assertEqual(enum, ["telegram"])
        self.assertNotIn("stickers", request["instructions"].lower())

    async def test_telegram_notice_is_preactivated_before_one_model_call(self):
        activations = []

        async def activate(spec, session):
            activations.append((spec.id, session.turn_id))
            return {
                "target_token": "turn-only-token",
                "messages": [{"id": 7, "text": "Привет"}],
            }

        agent, model, *_ = self._agent(
            [
                _final(_telegram_final("привет в ответ"))
            ],
            on_activate=activate,
            telegram_fast_enabled=True,
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            source_skill="telegram",
            occurred_at=datetime.now(timezone.utc),
            metadata=_production_notice_metadata(),
        )
        result = await agent.run_turn(trigger)

        self.assertEqual(result.active_skills, ("telegram",))
        self.assertEqual(activations, [("telegram", trigger.id)])
        self.assertEqual(len(model.responses.requests), 1)
        first = model.responses.requests[0]
        self.assertEqual(first["tools"], [])
        self.assertIn(
            "telegram",
            first["text"]["format"]["schema"]["properties"],
        )
        self.assertEqual(
            set(first["text"]["format"]["schema"]["properties"]),
            {"telegram"},
        )
        self.assertEqual(
            set(first["text"]["format"]["schema"]["required"]),
            {"telegram"},
        )
        self.assertEqual(
            first["text"]["format"]["schema"]["properties"]["telegram"]
            ["anyOf"][1]["properties"]["messages"]["maxItems"],
            1,
        )
        self.assertEqual(first["max_output_tokens"], 500)
        self.assertEqual(first["metadata"], {"agy_priority": "interactive"})
        self.assertIn("Привет", json.dumps(first["input"], ensure_ascii=False))
        self.assertIn("уже находишься в Telegram", first["instructions"])
        self.assertIn("Не вызывай никаких инструментов", first["instructions"])
        self.assertNotIn("Чтобы использовать внешний навык", first["instructions"])
        self.assertNotIn("Доступные закрытые навыки", first["instructions"])
        self.assertEqual(result.tool_results, ())
        self.assertEqual(result.model_rounds, 1)
        self.assertIsNotNone(result.model_elapsed_ms)
        self.assertIn("state_update", result.payload)
        self.assertEqual(result.payload["relationship_updates"], [])
        self.assertNotIn("memory_note", result.payload)

    def test_materialized_sticker_classifier_is_conservative_and_bilingual(self):
        sticker_requests = (
            "Пришли мне стикер, пожалуйста",
            "Please send me a sticker",
        )
        ordinary = (
            "Напомни мне через час проверить духовку",
            "Could you remind me tomorrow?",
            "Разбуди меня в 7 утра",
            "Wake me up at 7",
            "Отправь мне это сообщение завтра вечером",
            "Text me later",
            "Запланируй сообщение на завтра",
            "Schedule this message for tomorrow",
            "Вчера она отправила стикер и поставила напоминание.",
            "I was reminded of a sticker from yesterday.",
            "Как вообще устроены напоминания?",
            "My schedule is busy today.",
            "Просто поболтаем",
        )

        for text in sticker_requests:
            with self.subTest(text=text):
                self.assertTrue(
                    MilanaAgent._materialized_telegram_requests_sticker(
                        {"context": {"messages": [{"text": text}]}}
                    )
                )
        for text in ordinary:
            with self.subTest(text=text):
                self.assertFalse(
                    MilanaAgent._materialized_telegram_requests_sticker(
                        {"context": {"messages": [{"text": text}]}}
                    )
                )
        self.assertFalse(
            MilanaAgent._materialized_telegram_requests_sticker(
                {
                    "context": {
                        "messages": [{"text": "обычный текущий текст"}],
                        "history": [{"text": "Напомни мне через час"}],
                    }
                }
            )
        )

    async def test_materialized_sticker_command_gets_only_sticker_tools(self):
        async def activate(_spec, _session):
            return {
                "target_token": "turn-only-token",
                "messages": [{"message_id": 7, "text": "Пришли мне стикер"}],
            }

        agent, model, _, _, stickers = self._agent(
            [
                _function_call(
                    "open_sticker_picker", {"pack_id": None}, "picker"
                ),
                _function_call(
                    "send_sticker", {"sticker_id": "P001:S001"}, "send"
                ),
                _final({"telegram": None}),
            ],
            on_activate=activate,
            telegram_fast_enabled=True,
        )

        result = await agent.run_turn(
            TurnTrigger(
                kind="telegram_notice",
                source_skill="telegram",
                occurred_at=datetime.now(timezone.utc),
                metadata=_production_notice_metadata(),
            )
        )

        self.assertEqual(len(model.responses.requests), 3)
        for request in model.responses.requests[:2]:
            self.assertEqual(
                _tool_names(request),
                {"open_sticker_picker", "send_sticker", "schedule_sticker"},
            )
        self.assertNotIn(
            "state_update",
            model.responses.requests[0]["text"]["format"]["schema"]["properties"],
        )
        self.assertIn("Правила единственного", str(model.responses.requests[0]["input"]))
        self.assertNotIn("open_skill", _tool_names(model.responses.requests[0]))
        self.assertNotIn("schedule_message", _tool_names(model.responses.requests[0]))
        self.assertEqual(stickers.calls[-1][1], "send_sticker")
        self.assertTrue(result.trigger.metadata["_telegram_sticker_tools"])
        self.assertEqual(result.active_skills, ("telegram", "telegram.stickers"))

    async def test_sticker_route_rejects_hallucinated_non_sticker_tool(self):
        async def activate(_spec, _session):
            return {
                "target_token": "turn-only-token",
                "messages": [{"message_id": 7, "text": "пришли стикер"}],
            }

        agent, model, core, telegram, stickers = self._agent(
            [
                _function_call(
                    "write_diary", {"entry": "нельзя выполнять"}, "forbidden"
                ),
                _final(_telegram_final("лучше отвечу текстом")),
            ],
            on_activate=activate,
            telegram_fast_enabled=True,
        )

        result = await agent.run_turn(
            TurnTrigger(
                kind="telegram_notice",
                occurred_at=datetime.now(timezone.utc),
                metadata=_production_notice_metadata(),
            )
        )

        self.assertEqual(core.calls, [])
        self.assertEqual(telegram.calls, [])
        self.assertEqual(stickers.calls, [])
        self.assertEqual(len(result.tool_results), 1)
        self.assertFalse(result.tool_results[0].ok)
        self.assertNotIn("write_diary", _tool_names(model.responses.requests[0]))

    async def test_media_notice_is_direct_without_tools_and_outside_text_sla(self):
        async def activate(_spec, _session):
            return {
                "target_token": "turn-only-token",
                "messages": [
                    {
                        "message_id": 7,
                        "text": "что на фото?",
                        "media_type": "photo",
                        "media_path": "runtime/photo.png",
                    }
                ],
            }

        metadata = _production_notice_metadata()
        metadata["notices"][0]["media_type"] = "photo"
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=datetime.now(timezone.utc),
            metadata=metadata,
        )
        agent, model, *_ = self._agent(
            [_final(_telegram_final("вижу фото"))],
            on_activate=activate,
            tool_result_content=lambda _result: [
                {"type": "input_image", "image_url": "data:image/png;base64,eA=="}
            ],
            telegram_fast_enabled=True,
        )

        await agent.run_turn(trigger)

        self.assertEqual(len(model.responses.requests), 1)
        self.assertEqual(model.responses.requests[0]["tools"], [])
        self.assertNotIn(
            "state_update",
            model.responses.requests[0]["text"]["format"]["schema"]["properties"],
        )
        self.assertTrue(agent._is_direct_telegram_trigger(trigger))
        self.assertFalse(agent._is_fast_telegram_trigger(trigger))

    async def test_materialized_reminder_command_is_direct_without_tools(self):
        async def activate(_spec, _session):
            return {
                "target_token": "turn-only-token",
                "messages": [
                    {"message_id": 7, "text": "Remind me in an hour to stretch"}
                ],
            }

        agent, model, _, telegram, _ = self._agent(
            [
                _final(_telegram_final("i can't set that from this chat")),
            ],
            on_activate=activate,
            telegram_fast_enabled=True,
        )

        result = await agent.run_turn(
            TurnTrigger(
                kind="telegram_notice",
                source_skill="telegram",
                occurred_at=datetime.now(timezone.utc),
                metadata=_production_notice_metadata(),
            )
        )

        self.assertEqual(len(model.responses.requests), 1)
        self.assertEqual(model.responses.requests[0]["tools"], [])
        self.assertEqual(telegram.calls, [])
        self.assertNotIn("_telegram_sticker_tools", result.trigger.metadata)

    async def test_notice_activation_failure_stops_before_model_call(self):
        async def deny(_spec, _session):
            raise PermissionError("host denied activation")

        agent, model, *_ = self._agent(
            [], on_activate=deny, telegram_fast_enabled=True
        )
        with self.assertRaisesRegex(PermissionError, "host denied activation"):
            await agent.run_turn(
                TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=datetime.now(timezone.utc),
                )
            )
        self.assertEqual(model.responses.requests, [])

    async def test_provider_queue_metadata_is_exposed_on_turn_result(self):
        response = _final(_telegram_final())
        response.agy_queue_wait_ms = 17.5
        response.agy_model_ms = 23.25
        response.agy_model_calls = 2
        agent, *_ = self._agent([response], telegram_fast_enabled=True)

        result = await agent.run_turn(
            TurnTrigger(
                kind="telegram_notice",
                occurred_at=datetime.now(timezone.utc),
            )
        )

        self.assertEqual(result.model_rounds, 2)
        self.assertEqual(result.model_elapsed_ms, 23.25)
        self.assertEqual(result.provider_queue_ms, 17.5)

    async def test_disabled_fast_path_keeps_legacy_activation_schema_and_limits(self):
        activations = []

        async def activate(spec, _session):
            activations.append(spec.id)
            return {"target_token": "turn-only-token", "messages": []}

        legacy = empty_turn_payload(telegram=True)
        legacy["telegram"] = {
            "target_token": "turn-only-token",
            "messages": ["обычный ответ"],
            "reaction": None,
            "blacklist_sender": False,
        }
        agent, model, *_ = self._agent(
            [
                _function_call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(legacy),
            ],
            on_activate=activate,
            telegram_fast_enabled=False,
        )

        result = await agent.run_turn(
            TurnTrigger(
                kind="telegram_notice",
                occurred_at=datetime.now(timezone.utc),
                metadata={"chat_id": 77},
            )
        )

        self.assertEqual(activations, ["telegram"])
        self.assertEqual(len(model.responses.requests), 2)
        first, second = model.responses.requests
        self.assertIn("open_skill", _tool_names(first))
        self.assertEqual(first["max_output_tokens"], 1200)
        self.assertIn("Чтобы использовать внешний навык", first["instructions"])
        self.assertNotIn("telegram", first["text"]["format"]["schema"]["properties"])
        self.assertIn("state_update", second["text"]["format"]["schema"]["properties"])
        self.assertEqual(result.payload, legacy)

    async def test_fast_tool_call_is_not_executed_or_regenerated(self):
        agent, model, _, telegram, _ = self._agent(
            [_function_call("schedule_message", {"message": "потом"}, "call")],
            telegram_fast_enabled=True,
        )

        with self.assertRaisesRegex(ValueError, "без tool calls"):
            await agent.run_turn(
                TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=datetime.now(timezone.utc),
                    metadata={"chat_id": 77},
                )
            )

        self.assertEqual(len(model.responses.requests), 1)
        self.assertEqual(model.responses.requests[0]["tools"], [])
        # Trusted activation is handled by the registry; the hallucinated
        # channel action never reaches the Telegram executor.
        self.assertEqual(telegram.calls, [])

    async def test_direct_route_rejects_full_legacy_world_payload(self):
        legacy = empty_turn_payload(telegram=True)
        legacy["telegram"] = {
            "target_token": "turn-only-token",
            "messages": ["legacy"],
            "reaction": None,
            "blacklist_sender": False,
        }
        agent, model, *_ = self._agent(
            [_final(legacy)], telegram_fast_enabled=True
        )

        with self.assertRaisesRegex(ValueError, "Final payload fields"):
            await agent.run_turn(
                TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=datetime.now(timezone.utc),
                    metadata={"chat_id": 77},
                )
            )

        self.assertEqual(len(model.responses.requests), 1)

    async def test_direct_route_rejects_legacy_world_fields(self):
        invalid = _telegram_final()
        invalid["relationship_delta"] = {
            "closeness": 1,
            "reciprocity": 0,
            "tension": 0,
        }
        agent, model, *_ = self._agent(
            [_final(invalid)], telegram_fast_enabled=True
        )

        with self.assertRaisesRegex(ValueError, "Final payload fields"):
            await agent.run_turn(
                TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=datetime.now(timezone.utc),
                    metadata={"chat_id": 77},
                )
            )

        self.assertEqual(len(model.responses.requests), 1)

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
