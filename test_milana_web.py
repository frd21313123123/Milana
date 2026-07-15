import json
import unittest
import urllib.request

from milana_state import MilanaStateStore
from milana_web import start_web_server


class EmbeddedWebPanelTests(unittest.TestCase):
    def setUp(self):
        self.state = MilanaStateStore()
        self.actions = []
        self.service_status = {
            "telegram_host": {"connected": True, "process_running": True},
            "pending_replies": [
                {
                    "chat_id": "77",
                    "status": "waiting",
                    "message_count": 2,
                    "respond_at": "2026-07-15T16:30:00+05:00",
                    "detail": "ответ через 1–4 мин",
                }
            ],
            "next_reply": {
                "chat_id": "77",
                "status": "waiting",
                "message_count": 2,
                "respond_at": "2026-07-15T16:30:00+05:00",
                "detail": "ответ через 1–4 мин",
            },
            "telegram_latency": {
                "enabled": False,
                "sample_size": 0,
                # A stale/legacy producer value must not be presented as a
                # successful rollout by the web API or panel.
                "slo_met": True,
                "phases": {
                    "context_ms": {"p95": None},
                    "provider_queue_ms": {"p95": None},
                    "model_ms": {"p95": None},
                    "send_ms": {"p95": None},
                },
            },
        }
        self.panel = start_web_server(
            port=0,
            state_store=self.state,
            callbacks={
                "wake_now": lambda: self.actions.append("wake"),
                "update_state": lambda body: self.actions.append(("state", body)),
            },
            status_provider=lambda: self.service_status,
        )

    def tearDown(self):
        self.panel.stop()
        self.state.close()

    def test_status_exposes_life_and_host_state(self):
        with urllib.request.urlopen(self.panel.url + "api/status", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertIn("life", payload)
        self.assertEqual(payload["life"]["needs"]["social"], 50)
        self.assertTrue(payload["service"]["telegram_host"]["connected"])
        self.assertEqual(payload["service"]["next_reply"]["chat_id"], "77")
        self.assertEqual(
            payload["service"]["pending_replies"][0]["message_count"], 2
        )
        self.assertIsNone(payload["service"]["telegram_latency"]["slo_met"])
        self.assertTrue(payload["schedule"]["available"])
        self.assertIn("activities", payload["schedule"])
        self.assertIn("response_policy", payload["schedule"])

    def test_status_preserves_full_active_latency_distribution(self):
        self.service_status["telegram_latency"] = {
            "enabled": True,
            "sample_size": 50,
            "p50_ms": 4_100.0,
            "p95_ms": 9_500.0,
            "p99_ms": 12_000.0,
            "breaches": 2,
            "exceed_rate": 0.04,
            "slo_met": True,
            "llm_calls": {"total": 50, "average": 1.0, "p95": 1.0},
            "phases": {
                "context_ms": {"p50": 20.0, "p95": 40.0, "p99": 60.0},
                "provider_queue_ms": {"p50": 0.0, "p95": 10.0, "p99": 20.0},
                "model_ms": {"p50": 3_900.0, "p95": 9_100.0, "p99": 11_500.0},
                "send_ms": {"p50": 180.0, "p95": 350.0, "p99": 420.0},
            },
        }
        with urllib.request.urlopen(self.panel.url + "api/status", timeout=10) as response:
            latency = json.loads(response.read().decode("utf-8"))["service"][
                "telegram_latency"
            ]
        self.assertTrue(latency["slo_met"])
        self.assertEqual(latency["p95_ms"], 9_500.0)
        self.assertEqual(latency["llm_calls"]["average"], 1.0)
        self.assertEqual(latency["phases"]["model_ms"]["p99"], 11_500.0)

    def test_status_never_preserves_green_sla_for_incomplete_or_unknown_window(self):
        self.service_status["telegram_latency"] = {
            "enabled": True,
            "sample_size": 1,
            "ordinary_text_turns": 5,
            "delivery_rate": 0.2,
            "missing_samples": 4,
            "eligibility_unknown": 0,
            "completeness": {
                "expected_samples": 5,
                "measured_samples": 1,
                "missing_samples": 4,
                "unknown_eligibility": 0,
                "rate": 0.2,
                "complete": False,
            },
            "slo_met": True,
        }
        with urllib.request.urlopen(self.panel.url + "api/status", timeout=10) as response:
            latency = json.loads(response.read().decode("utf-8"))["service"][
                "telegram_latency"
            ]
        self.assertFalse(latency["slo_met"])
        self.assertEqual(latency["delivery_rate"], 0.2)

        self.service_status["telegram_latency"]["missing_samples"] = 0
        self.service_status["telegram_latency"]["completeness"][
            "missing_samples"
        ] = 0
        self.service_status["telegram_latency"]["eligibility_unknown"] = 1
        self.service_status["telegram_latency"]["completeness"][
            "unknown_eligibility"
        ] = 1
        self.service_status["telegram_latency"]["slo_met"] = True
        with urllib.request.urlopen(self.panel.url + "api/status", timeout=10) as response:
            latency = json.loads(response.read().decode("utf-8"))["service"][
                "telegram_latency"
            ]
        self.assertIsNone(latency["slo_met"])

    def test_management_action_uses_embedded_callback(self):
        request = urllib.request.Request(
            self.panel.url + "api/heartbeat/wake",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(self.actions, ["wake"])

    def test_management_body_is_forwarded_to_callback(self):
        request = urllib.request.Request(
            self.panel.url + "api/state/update",
            data=json.dumps({"mood": "радостное"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(self.actions, [("state", {"mood": "радостное"})])

    def test_panel_has_no_external_runtime_dependencies(self):
        with urllib.request.urlopen(self.panel.url, timeout=10) as response:
            html = response.read().decode("utf-8")
        self.assertIn("Мир и цели", html)
        self.assertIn("/api/relationships/update", html)
        self.assertIn('id="telegram-latency"', html)
        self.assertIn('id="telegram-phases"', html)
        self.assertIn('id="next-reply"', html)
        self.assertIn('id="schedule-list"', html)
        self.assertIn("pending_replies", html)
        self.assertIn("replyEta", html)
        self.assertIn("l.enabled===true", html)
        self.assertIn("fast path выключен", html)
        self.assertIn("provider_queue_ms", html)
        self.assertNotIn("cdn.tailwindcss.com", html)
        self.assertNotIn("fonts.googleapis.com", html)


if __name__ == "__main__":
    unittest.main()
