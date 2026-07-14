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
            callbacks={"wake_now": lambda: self.actions.append("wake")},
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


if __name__ == "__main__":
    unittest.main()
