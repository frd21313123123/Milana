import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_telegram_fast_path import run_benchmark
from milana_state import MilanaStateStore
from telegram_client import (
    AIConfig,
    GEMINI_LLM_CHOICE,
    MessageFlowConfig,
    TelegramFastPathConfig,
)


class _FakeResponses:
    def __init__(self):
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        payload = {
            "memory_note": None,
            "relationship_delta": None,
            "telegram": {
                "target_token": "benchmark-target-token",
                "messages": ["короткий ответ"],
                "reaction": None,
                "blacklist_sender": False,
            },
        }
        return SimpleNamespace(
            output=[],
            output_text=json.dumps(payload, ensure_ascii=False),
            agy_queue_wait_ms=0.25,
            agy_model_ms=1.0,
            agy_total_ms=1.25,
            agy_model_calls=1,
        )


class _FakeModel:
    def __init__(self):
        self.responses = _FakeResponses()


class TelegramFastPathBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_model_smoke_never_needs_telegram(self):
        model = _FakeModel()
        config = AIConfig(
            api_key="",
            model="fake-gemini",
            instructions="Отвечай кратко по-русски.",
            temperature=0.0,
            max_output_tokens=500,
            message_flow=MessageFlowConfig(),
            telegram_fast_path=TelegramFastPathConfig(),
            provider=GEMINI_LLM_CHOICE,
        )

        report = await run_benchmark(runs=2, config=config, model_client=model)

        self.assertEqual(report["runs"], 2)
        self.assertEqual(report["success"], 2)
        self.assertEqual(report["receipts"], 2)
        self.assertEqual(report["failures"], 0)
        self.assertEqual(report["model_calls"]["known_total"], 2)
        self.assertEqual(report["model_calls"]["mean"], 1.0)
        self.assertEqual(report["sla"]["sample_size"], 2)
        self.assertTrue(report["sla"]["met"])
        self.assertEqual(report["fake_telegram_host"]["sent_messages"], 2)
        self.assertEqual(report["fake_telegram_host"]["receipts"], 2)
        self.assertEqual(len(model.responses.requests), 2)
        self.assertEqual(model.responses.requests[0]["tools"], [])
        self.assertIn("tg:7700000001:1", model.responses.requests[0]["input"][0]["content"])

    async def test_metric_write_loss_cannot_report_sla_success(self):
        model = _FakeModel()
        config = AIConfig(
            api_key="",
            model="fake-gemini",
            instructions="Отвечай кратко по-русски.",
            temperature=0.0,
            max_output_tokens=500,
            message_flow=MessageFlowConfig(),
            telegram_fast_path=TelegramFastPathConfig(),
            provider=GEMINI_LLM_CHOICE,
        )
        original = MilanaStateStore.record_telegram_turn_metric
        writes = 0

        def lose_after_first(store, metric):
            nonlocal writes
            writes += 1
            if writes > 1:
                raise RuntimeError("simulated metric persistence loss")
            return original(store, metric)

        with patch.object(
            MilanaStateStore,
            "record_telegram_turn_metric",
            new=lose_after_first,
        ):
            report = await run_benchmark(runs=2, config=config, model_client=model)

        self.assertEqual(report["success"], 2)
        self.assertEqual(report["receipts"], 2)
        self.assertEqual(report["model_calls"]["mean"], 1.0)
        self.assertEqual(report["sla"]["sample_size"], 1)
        self.assertFalse(report["sla"]["checks"]["sample_count"])
        self.assertFalse(report["sla"]["met"])


if __name__ == "__main__":
    unittest.main()
