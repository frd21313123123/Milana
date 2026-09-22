import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from agy_recovery import (
    ensure_agy_proxy_patch,
    extract_unlocker_key,
    extract_unlocker_message_ids,
    installed_unlocker_version,
    is_unlocker_recoverable_error,
    prefer_unlocker_relay,
    restart_unlocker_proxy,
)


class AgyRecoveryTests(unittest.TestCase):
    def test_version_banner_accepts_release_with_multiple_segments(self) -> None:
        completed = MagicMock(stdout="Antigravity version: 2.15.1.2", stderr="")
        with patch("agy_recovery.subprocess.run", return_value=completed):
            self.assertEqual(
                installed_unlocker_version(Path("/tmp/ag_unlocker")), "2.15.1.2"
            )

    def test_key_is_accepted_only_next_to_matching_version(self) -> None:
        key = "A" * 24
        messages = [
            {"text": "Ключ для v2.15.1 +"},
            {"text": key},
        ]
        self.assertEqual(extract_unlocker_key(messages, "2.15.1"), key)
        self.assertIsNone(extract_unlocker_key(messages, "2.15.0"))

    def test_pinned_channel_links_are_deduplicated(self) -> None:
        messages = [
            {"text": "https://t.me/nova_txt/69864/154802"},
            {"text": "https://t.me/nova_txt/154802"},
            {"text": "https://t.me/nova_txt/154803"},
        ]
        self.assertEqual(
            extract_unlocker_message_ids(messages), (154802, 154803)
        )

    def test_region_error_detection_excludes_generic_failures(self) -> None:
        self.assertTrue(
            is_unlocker_recoverable_error(
                RuntimeError("User location is not supported for the API use")
            )
        )
        self.assertFalse(is_unlocker_recoverable_error(RuntimeError("timeout")))

    def test_relay_settings_preserve_unrelated_options(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory, "settings.json")
            path.write_text(
                json.dumps(
                    {
                        "builtin_exits": True,
                        "local_proxy": False,
                        "own_proxy_enabled": True,
                        "verify_tls": True,
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(prefer_unlocker_relay(path))

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(saved["builtin_exits"])
            self.assertTrue(saved["local_proxy"])
            self.assertFalse(saved["own_proxy_enabled"])
            self.assertTrue(saved["verify_tls"])

    def test_agy_proxy_patch_is_atomic_idempotent_and_backed_up(self) -> None:
        with TemporaryDirectory() as directory:
            executable = Path(directory, "agy")
            original = (
                b"prefix-ineligible-https_proxy-middle-https_proxy-suffix"
            )
            executable.write_bytes(original)

            self.assertTrue(ensure_agy_proxy_patch(executable))
            self.assertTrue(ensure_agy_proxy_patch(executable))

            self.assertEqual(
                executable.read_bytes(),
                b"prefix-inexigible-AG_LS_PROXY-middle-AG_LS_PROXY-suffix",
            )
            self.assertEqual(
                Path(directory, "agy.before-ag-ls-proxy").read_bytes(), original
            )

    @unittest.skipIf(os.name == "nt", "systemd proxy runs only on Linux")
    def test_restart_waits_for_proxy_listener(self) -> None:
        completed = MagicMock(returncode=0)
        connection = MagicMock()
        connection.__enter__.return_value = connection
        with (
            TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "AGY_EXECUTABLE": str(Path(directory, "agy")),
                    "AG_UNLOCKER_SETTINGS": str(Path(directory, "settings.json")),
                },
            ),
            patch("agy_recovery.subprocess.run", return_value=completed) as run,
            patch("agy_recovery.socket.create_connection", return_value=connection),
        ):
            Path(directory, "agy").write_bytes(b"ineligible-https_proxy")
            self.assertTrue(restart_unlocker_proxy())

        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
