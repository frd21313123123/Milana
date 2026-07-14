import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from milana.host_supervisor import (
    RESTART_BACKOFF_SECONDS,
    SkillHostSupervisor,
    restart_delay,
)
from milana_ipc import JsonRpcServer


class HostSupervisorPolicyTests(unittest.TestCase):
    def test_restart_backoff_sequence_and_cap(self):
        self.assertEqual(
            [restart_delay(index) for index in range(8)],
            [1, 2, 5, 10, 30, 60, 60, 60],
        )
        self.assertEqual(RESTART_BACKOFF_SECONDS, (1, 2, 5, 10, 30, 60))

    def test_restart_attempt_is_validated(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                restart_delay(invalid)


class _FakeProcess:
    def __init__(self, *, crash=False):
        self.returncode = None
        self._done = asyncio.Event()
        if crash:
            self.returncode = 1
            self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self._done.set()

    def kill(self):
        self.terminate()


class HostSupervisorRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_crashed_child_is_restarted(self):
        server = JsonRpcServer("secret")
        await server.start()
        temporary = TemporaryDirectory()
        root = Path(temporary.name)
        token = root / "token"
        token.write_text("secret", encoding="utf-8")
        media = root / "media"
        media.mkdir()
        processes = []

        async def factory(_command):
            process = _FakeProcess(crash=not processes)
            processes.append(process)
            return process

        supervisor = SkillHostSupervisor(
            server,
            token_file=token,
            runtime_dir=media,
            process_factory=factory,
        )
        try:
            with patch("milana.host_supervisor.restart_delay", return_value=0.01):
                await supervisor.start()
                for _ in range(100):
                    if len(processes) >= 2:
                        break
                    await asyncio.sleep(0.01)
            self.assertGreaterEqual(len(processes), 2)
            self.assertTrue(supervisor.process_running)
        finally:
            await supervisor.stop()
            await server.close()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
