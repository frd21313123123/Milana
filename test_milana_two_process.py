import asyncio
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from milana import empty_turn_payload
from milana.host_supervisor import SkillHostSupervisor
from milana_ipc import JsonRpcServer, load_or_create_auth_token
from milana_memory import MilanaMemoryStore
from milana_schedule import load_routine
from milana_service import MilanaService
from milana_state import MilanaStateStore
from telegram_client import AIConfig, MessageFlowConfig


NOW = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)


def _call(name, arguments, call_id):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments),
                call_id=call_id,
            )
        ],
        output_text="",
    )


def _final(payload):
    return SimpleNamespace(output=[], output_text=json.dumps(payload, ensure_ascii=False))


class _Responses:
    def __init__(self, values):
        self.values = list(values)

    async def create(self, **_request):
        return self.values.pop(0)


class TwoProcessMilanaTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _host_script() -> Path:
        return (
            Path(__file__).resolve().parent
            / "test_support"
            / "fake_telegram_skill_host.py"
        )

    async def test_notice_telegram_stickers_reply_crosses_real_ipc_process(self):
        temporary = TemporaryDirectory()
        root = Path(temporary.name)
        runtime = root / "runtime"
        runtime.mkdir()
        seed = {
            "notice_id": "tg:77:9",
            "chat_id": 77,
            "message_id": 9,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
            "text": "привет, пришли стикер из дочернего процесса",
        }
        (runtime / "fake-notice.json").write_text(
            json.dumps(seed, ensure_ascii=False), encoding="utf-8"
        )
        token_file = root / "host.token"
        token = load_or_create_auth_token(token_file)
        server = JsonRpcServer(token, request_timeout=10.0)
        await server.start()
        supervisor = SkillHostSupervisor(
            server,
            token_file=token_file,
            runtime_dir=runtime,
            host_script=self._host_script(),
            python_executable=sys.executable,
            dev_mode=True,
        )
        database = root / "milana.sqlite3"
        memory = MilanaMemoryStore(database)
        state = MilanaStateStore(database)
        final = {
            "telegram": {
                "target_token": "fake-target-token",
                "messages": ["привет"],
                "reaction": None,
                "blacklist_sender": False,
            }
        }
        model = SimpleNamespace(
            responses=_Responses(
                [
                    _call("open_sticker_picker", {"pack_id": None}, "index"),
                    _call("open_sticker_picker", {"pack_id": "P001"}, "pack"),
                    _call("send_sticker", {"sticker_id": "P001:S001"}, "send"),
                    _final(final),
                ]
            )
        )
        config = AIConfig(
            api_key="test",
            model="fake",
            instructions="персона для изолированного теста",
            temperature=0.7,
            max_output_tokens=1200,
            message_flow=MessageFlowConfig(
                input_quiet_seconds=0,
                input_max_wait_seconds=0,
                max_reply_messages=5,
                inter_message_min_delay_seconds=0,
                inter_message_max_delay_seconds=0,
            ),
        )
        service = MilanaService(
            config=config,
            model_client=model,
            memory=memory,
            state=state,
            routine=load_routine(),
            rpc_server=server,
            supervisor=supervisor,
            dev_mode=True,
            now=lambda: NOW,
        )
        server.register_method("telegram.notice", service._rpc_telegram_notice)
        try:
            await service.start(web_port=None)
            log_path = runtime / "fake-actions.jsonl"
            for _ in range(200):
                if log_path.exists() and "acknowledge" in log_path.read_text(
                    encoding="utf-8"
                ):
                    break
                await asyncio.sleep(0.025)
            else:
                self.fail(f"subprocess flow did not finish: {service.last_turn_error}")

            actions = [
                json.loads(line)["action"]
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                actions,
                [
                    "typing",
                    "open_sticker_picker",
                    "open_sticker_picker",
                    "send_sticker",
                    "send_messages",
                    "acknowledge",
                ],
            )
            history = memory.get_chat_history(77)
            self.assertEqual([item.role for item in history], ["user", "assistant"])
        finally:
            await service.stop()
            temporary.cleanup()

    async def test_real_child_process_is_reauthenticated_after_one_crash(self):
        temporary = TemporaryDirectory()
        root = Path(temporary.name)
        runtime = root / "runtime"
        runtime.mkdir()
        seed = {
            "notice_id": "tg:1:1",
            "chat_id": 1,
            "message_id": 1,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 2, "display_name": "Лера"},
            "media_type": "text",
            "text": "restart probe",
        }
        (runtime / "fake-notice.json").write_text(
            json.dumps(seed, ensure_ascii=False), encoding="utf-8"
        )
        (runtime / "fake-crash-once").write_text("1", encoding="utf-8")
        token_file = root / "host.token"
        token = load_or_create_auth_token(token_file)
        server = JsonRpcServer(token, request_timeout=5.0)

        async def accept_notice(_params, _request):
            return {"accepted": True}

        server.register_method("telegram.notice", accept_notice)
        await server.start()
        supervisor = SkillHostSupervisor(
            server,
            token_file=token_file,
            runtime_dir=runtime,
            host_script=self._host_script(),
            python_executable=sys.executable,
            dev_mode=True,
        )
        try:
            await supervisor.start()
            starts_path = runtime / "fake-starts.txt"
            for _ in range(400):
                starts = (
                    starts_path.read_text(encoding="utf-8").splitlines()
                    if starts_path.exists()
                    else []
                )
                if len(starts) >= 2 and supervisor.connected:
                    break
                await asyncio.sleep(0.025)
            else:
                self.fail(f"host was not restarted: {supervisor.status()}")
            self.assertTrue((runtime / "fake-crashed").exists())
            self.assertTrue(supervisor.process_running)
            self.assertEqual(supervisor.last_exit_code, 0)
        finally:
            await supervisor.stop()
            await server.close()
            temporary.cleanup()

    async def test_persisted_delayed_message_is_sent_after_service_reopen(self):
        temporary = TemporaryDirectory()
        root = Path(temporary.name)
        runtime = root / "runtime"
        runtime.mkdir()
        seed = {
            "notice_id": "tg:77:9",
            "chat_id": 77,
            "message_id": 9,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
            "text": "unused",
        }
        (runtime / "fake-notice.json").write_text(
            json.dumps(seed, ensure_ascii=False), encoding="utf-8"
        )
        (runtime / "fake-disable-notice").write_text("1", encoding="utf-8")
        token_file = root / "host.token"
        token = load_or_create_auth_token(token_file)
        server = JsonRpcServer(token, request_timeout=10.0)
        await server.start()
        supervisor = SkillHostSupervisor(
            server,
            token_file=token_file,
            runtime_dir=runtime,
            host_script=self._host_script(),
            python_executable=sys.executable,
            dev_mode=True,
        )
        database = root / "milana.sqlite3"
        before_restart = MilanaMemoryStore(database)
        scheduled = before_restart.schedule_pulse_message(
            77, "я не потерялась", due_at=NOW
        )
        before_restart.close()
        memory = MilanaMemoryStore(database)
        state = MilanaStateStore(database)
        config = AIConfig(
            api_key="test",
            model="fake",
            instructions="персона для изолированного теста",
            temperature=0.7,
            max_output_tokens=1200,
            message_flow=MessageFlowConfig(
                input_quiet_seconds=0,
                input_max_wait_seconds=0,
                max_reply_messages=5,
                inter_message_min_delay_seconds=0,
                inter_message_max_delay_seconds=0,
            ),
        )
        service = MilanaService(
            config=config,
            model_client=SimpleNamespace(
                responses=_Responses([_final(empty_turn_payload())])
            ),
            memory=memory,
            state=state,
            routine=load_routine(),
            rpc_server=server,
            supervisor=supervisor,
            dev_mode=True,
            now=lambda: NOW,
        )
        try:
            await service.start(web_port=None)
            log_path = runtime / "fake-actions.jsonl"
            for _ in range(300):
                actions = (
                    [
                        json.loads(line)
                        for line in log_path.read_text(encoding="utf-8").splitlines()
                    ]
                    if log_path.exists()
                    else []
                )
                if any(item["action"] == "send_messages" for item in actions):
                    break
                await asyncio.sleep(0.025)
            else:
                self.fail(f"delayed send did not run: {service.last_turn_error}")
            sent = next(item for item in actions if item["action"] == "send_messages")
            self.assertEqual(sent["payload"]["arguments"]["messages"], ["я не потерялась"])
            self.assertEqual(
                sent["payload"]["idempotency_key"], f"delayed:{scheduled.id}"
            )
            for _ in range(100):
                if memory.get_pulse_tasks()[0].status == "completed":
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(memory.get_pulse_tasks()[0].status, "completed")
        finally:
            await service.stop()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
