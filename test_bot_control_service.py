import unittest
from pathlib import Path


class BotControlServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (Path(__file__).parent / "bot_control.bat").read_text(
            encoding="utf-8"
        )

    def test_controller_launches_milana_service_not_legacy_ai_bot(self):
        self.assertIn('set "SCRIPT=%ROOT%milana_service.py"', self.text)
        self.assertNotIn("'%SCRIPT%', 'ai-bot'", self.text)
        self.assertIn("$env:SCRIPT", self.text)
        self.assertIn("CommandLine.IndexOf($script", self.text)

    def test_web_command_only_opens_embedded_panel(self):
        block = self.text.split(":open_web", 1)[1].split(":read_pid", 1)[0]
        self.assertIn("web panel is embedded", block.lower())
        self.assertNotIn("milana_web.py", block)
        self.assertNotIn("--no-browser", block)

    def test_legacy_cli_is_documented_as_service_alias_in_code(self):
        telegram = (Path(__file__).parent / "telegram_client.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('if args.command == "ai-bot":', telegram)
        self.assertIn("milana_service_main", telegram)

    def test_saved_pid_is_bound_to_the_original_process_instance(self):
        self.assertIn("BOT_START_TICKS", self.text)
        self.assertIn("ToFileTimeUtc", self.text)
        self.assertIn("$env:SCRIPT", self.text)
        saved_pid_block = self.text.split("if exist \"%PID_FILE%\" (", 1)[1]
        saved_pid_block = saved_pid_block.split("rem Fall back to discovery", 1)[0]
        self.assertIn("call :is_running !SAVED_PID! !SAVED_START_TICKS!", saved_pid_block)


if __name__ == "__main__":
    unittest.main()
