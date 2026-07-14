import json
import unittest
import urllib.request

from milana_state import MilanaStateStore
from milana_web import start_web_server


class EmbeddedWebPanelTests(unittest.TestCase):
    def setUp(self):
        self.state = MilanaStateStore()
        self.actions = []
        self.panel = start_web_server(
            port=0,
            state_store=self.state,
            callbacks={
                "wake_now": lambda: self.actions.append("wake"),
                "update_state": lambda body: self.actions.append(("state", body)),
            },
            status_provider=lambda: {
                "telegram_host": {"connected": True, "process_running": True}
            },
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
        self.assertNotIn("cdn.tailwindcss.com", html)
        self.assertNotIn("fonts.googleapis.com", html)


if __name__ == "__main__":
    unittest.main()
