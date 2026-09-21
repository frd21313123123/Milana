import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from milana import TurnTrigger
from milana_memory import MilanaMemoryStore
from milana_phone import PhoneSessionStore
from milana_schedule import load_routine
from milana_service import MilanaService
from milana_state import MilanaStateStore
from jev_provider import JevConfig, JevResult
from telegram_client import (
    AIConfig,
    MessageFlowConfig,
    PhoneSessionConfig,
    TelegramFastPathConfig,
    load_phone_session_config,
)


NOW = datetime(2026, 9, 21, 20, 0, tzinfo=timezone.utc)
SERVICE_NOW = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)


def _final(payload):
    return SimpleNamespace(output=[], output_text=json.dumps(payload, ensure_ascii=False))


class _Responses:
    def __init__(self, values):
        self.values = list(values)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        return self.values.pop(0)


class _JevPhone:
    def __init__(self):
        self.requests = []

    async def evaluate(self, **request):
        self.requests.append(request)
        return JevResult(
            model="jev-1.13.0",
            answers={
                "decision": {
                    "type": "choice",
                    "choice": "continue",
                    "confidence": 0.95,
                    "probabilities": {
                        "continue": 0.95,
                        "put_away": 0.03,
                        "go_to_sleep": 0.02,
                    },
                },
                "duration": {
                    "type": "choice",
                    "choice": "medium",
                    "confidence": 0.9,
                    "probabilities": {"short": 0.05, "medium": 0.9, "long": 0.05},
                },
                "dialog_0": {
                    "type": "choice",
                    "choice": "reply",
                    "confidence": 0.9,
                    "probabilities": {
                        "skip": 0.03,
                        "read": 0.03,
                        "reply": 0.9,
                        "react": 0.04,
                    },
                },
            },
            input_tokens=200,
            output_tokens=50,
            elapsed_ms=10,
        )

    async def record_rejection(self, _scenario, _exc):
        return None


class _Supervisor:
    def __init__(self):
        self.calls = []

    async def request(self, method, params, **options):
        self.calls.append((method, dict(params), dict(options)))
        if method == "telegram.list_dialogs":
            return {
                "dialogs": [
                    {
                        "target_ref": 77,
                        "title": "Лера",
                        "kind": "private",
                        "unread_count": 1,
                        "last_activity_at": SERVICE_NOW.isoformat(),
                        "actions": ["read", "reply", "react"],
                    }
                ],
                "has_more": False,
                "next_offset": None,
            }
        if method == "telegram.open":
            return {
                "turn_id": params["turn_id"],
                "target_token": "phone-token",
                "target_ref": 77,
                "messages": [
                    {
                        "message_id": 9,
                        "timestamp": SERVICE_NOW.isoformat(),
                        "sender": {"id": 88, "display_name": "Лера"},
                        "text": "привет",
                        "media_type": "text",
                    }
                ],
                "history": [],
            }
        if method == "telegram.execute":
            if params["action"] == "send_messages":
                return {
                    "status": "sent",
                    "sent_message_ids": [10],
                    "sent_part_indexes": [0],
                    "next_part_index": 1,
                    "total_parts": 1,
                }
            return {"status": "ok"}
        if method == "telegram.presence":
            return {"online": params["online"]}
        if method == "telegram.cleanup_turn":
            return {"cleaned": True}
        raise AssertionError(method)

    def status(self):
        return {"connected": True}


class PhoneSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.state = MilanaStateStore(Path(self.tmp.name) / "state.sqlite3")
        self.store = PhoneSessionStore(self.state)

    def tearDown(self):
        self.state.close()
        self.tmp.cleanup()

    def test_one_active_session_tracks_plan_and_results(self):
        session = self.store.start("telegram_notice", NOW, 240)
        self.assertEqual(self.store.start("spontaneous", NOW, 480).id, session.id)
        planned = self.store.apply_plan(
            session.id,
            {
                "decision": "continue",
                "reconsider_seconds": 360,
                "visits": [
                    {"target_ref": "10", "intent": "reply"},
                    {"target_ref": "20", "intent": "react"},
                ],
            },
            NOW,
        )
        self.assertEqual(planned.next_decision_at, NOW + timedelta(seconds=360))
        self.store.select_visit(session.id, "10")
        self.store.finish_visit(
            session.id,
            target_ref="10",
            intent="reply",
            outcome="replied",
            at=NOW,
        )
        snapshot = self.store.snapshot()
        self.assertEqual(snapshot["session"]["selected_chat"], None)
        self.assertEqual(
            snapshot["session"]["remaining_plan"],
            [{"target_ref": "20", "intent": "react"}],
        )
        self.assertEqual(snapshot["recent_actions"][-1]["outcome"], "replied")

    def test_restart_marks_active_session_interrupted(self):
        session = self.store.start("spontaneous", NOW, 120)
        reopened = PhoneSessionStore(self.state)
        reopened.recover(NOW + timedelta(seconds=10))
        snapshot = reopened.snapshot()
        self.assertEqual(snapshot["status"], "ended")
        self.assertEqual(snapshot["session"]["id"], session.id)
        self.assertEqual(snapshot["session"]["end_reason"], "interrupted")

    def test_plan_validation_rejects_out_of_bounds_window(self):
        with self.assertRaises(ValueError):
            self.store.validate_plan(
                {"decision": "continue", "reconsider_seconds": 481, "visits": []}
            )

    def test_dialog_read_ack_can_be_recovered_without_notice_ids(self):
        intent = self.state.prepare_telegram_ack_intent(
            "phone-read:10",
            10,
            [],
            [41, 42],
            at=NOW,
        )
        self.assertEqual(intent.notice_ids, ())
        self.assertEqual(intent.message_ids, (41, 42))
        self.assertEqual(
            self.state.list_pending_telegram_ack_intents()[0].action_key,
            "phone-read:10",
        )


class PhoneSessionConfigTests(unittest.TestCase):
    def test_old_config_enables_phone_sessions_with_defaults(self):
        config = load_phone_session_config({})
        self.assertTrue(config.enabled)
        self.assertEqual((config.duration_min_seconds, config.duration_max_seconds), (120, 480))

    def test_feature_can_be_disabled(self):
        self.assertFalse(
            load_phone_session_config({"phone_session": {"enabled": False}}).enabled
        )


class PhoneSessionServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        path = Path(self.tmp.name) / "service.sqlite3"
        self.memory = MilanaMemoryStore(path)
        self.state = MilanaStateStore(path)
        self.supervisor = _Supervisor()

    def tearDown(self):
        self.state.close()
        self.memory.close()
        self.tmp.cleanup()

    async def test_jev_phone_plan_avoids_main_model_when_confident(self):
        responses = _Responses([])
        jev = _JevPhone()
        config = AIConfig(
            api_key="test",
            model="fake",
            instructions="персона",
            temperature=0.7,
            max_output_tokens=1200,
            phone_session=PhoneSessionConfig(enabled=True),
            jev=JevConfig(enabled=True, api_key="test"),
        )
        service = MilanaService(
            config=config,
            model_client=SimpleNamespace(responses=responses),
            memory=self.memory,
            state=self.state,
            routine=load_routine(),
            rpc_server=SimpleNamespace(),
            supervisor=self.supervisor,
            jev_client=jev,
            dev_mode=False,
            now=lambda: SERVICE_NOW,
        )
        session = service.phone_sessions.start("test", SERVICE_NOW, 120)
        planned = await service._plan_phone_session(session, None)
        self.assertEqual(planned.remaining_plan[0].target_ref, "77")
        self.assertEqual(planned.remaining_plan[0].intent, "reply")
        self.assertEqual(
            planned.next_decision_at, SERVICE_NOW + timedelta(seconds=300)
        )
        self.assertEqual(len(jev.requests), 1)
        self.assertEqual(responses.requests, [])

    async def test_incoming_turn_plans_one_session_then_uses_existing_delivery(self):
        responses = _Responses(
            [
                _final(
                    {
                        "phone_session": {
                            "decision": "continue",
                            "reconsider_seconds": 240,
                            "visits": [{"target_ref": "77", "intent": "reply"}],
                        }
                    }
                ),
                _final(
                    {
                        "telegram": {
                            "target_token": "phone-token",
                            "messages": ["привет"],
                            "reaction": None,
                            "blacklist_sender": False,
                        }
                    }
                ),
            ]
        )
        config = AIConfig(
            api_key="test",
            model="fake",
            instructions="персона",
            temperature=0.7,
            max_output_tokens=1200,
            message_flow=MessageFlowConfig(
                input_quiet_seconds=0,
                input_max_wait_seconds=0,
                inter_message_min_delay_seconds=0,
                inter_message_max_delay_seconds=0,
            ),
            telegram_fast_path=TelegramFastPathConfig(dev_chat_only=False),
            phone_session=PhoneSessionConfig(enabled=True),
        )
        service = MilanaService(
            config=config,
            model_client=SimpleNamespace(responses=responses),
            memory=self.memory,
            state=self.state,
            routine=load_routine(),
            rpc_server=SimpleNamespace(),
            supervisor=self.supervisor,
            dev_mode=False,
            now=lambda: SERVICE_NOW,
        )
        notice_id = "tg:77:9"
        notice = {
            "source": "telegram",
            "notice_id": notice_id,
            "chat_id": 77,
            "message_id": 9,
            "timestamp": SERVICE_NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }
        self.state.record_telegram_notice(notice, received_at=SERVICE_NOW)
        result = await service._execute_turn(
            TurnTrigger(
                kind="telegram_notice",
                source_skill="telegram",
                occurred_at=SERVICE_NOW,
                revision=0,
                metadata={
                    "chat_id": 77,
                    "notice_ids": [notice_id],
                    "notices": [notice],
                },
            )
        )
        self.assertEqual(result.payload["telegram"]["messages"], ["привет"])
        snapshot = service.phone_sessions.snapshot()
        self.assertEqual(snapshot["status"], "active")
        self.assertEqual(snapshot["recent_actions"][-1]["outcome"], "replied")
        self.assertEqual(len(responses.requests), 2)
        self.assertEqual(
            [call[0] for call in self.supervisor.calls[:3]],
            ["telegram.presence", "telegram.list_dialogs", "telegram.open"],
        )
        self.assertTrue(
            any(call[0] == "telegram.execute" and call[1]["action"] == "acknowledge"
                for call in self.supervisor.calls)
        )


if __name__ == "__main__":
    unittest.main()
