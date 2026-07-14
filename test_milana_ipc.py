import asyncio
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from milana_ipc import (
    AuthenticationError,
    FrameTooLargeError,
    JsonRpcError,
    JsonRpcServer,
    MediaPathError,
    MediaPathValidator,
    connect_json_rpc,
    encode_frame,
    read_frame,
)


class FramingTests(unittest.IsolatedAsyncioTestCase):
    async def test_frame_uses_big_endian_length_and_round_trips_unicode(self) -> None:
        message = {"jsonrpc": "2.0", "method": "привет", "params": [1, 2]}
        encoded = encode_frame(message)
        self.assertEqual(struct.unpack(">I", encoded[:4])[0], len(encoded) - 4)

        reader = asyncio.StreamReader()
        reader.feed_data(encoded)
        reader.feed_eof()
        self.assertEqual(await read_frame(reader), message)

    async def test_oversized_incoming_and_outgoing_frames_are_rejected(self) -> None:
        with self.assertRaises(FrameTooLargeError):
            encode_frame({"data": "x" * 100}, max_frame=20)

        reader = asyncio.StreamReader()
        reader.feed_data(struct.pack(">I", 21))
        reader.feed_eof()
        with self.assertRaises(FrameTooLargeError):
            await read_frame(reader, max_frame=20)


class JsonRpcTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.servers: list[JsonRpcServer] = []
        self.clients = []

    async def asyncTearDown(self) -> None:
        await asyncio.gather(
            *(client.close() for client in self.clients), return_exceptions=True
        )
        await asyncio.gather(
            *(server.close() for server in self.servers), return_exceptions=True
        )

    async def make_server(self, token: str, **kwargs) -> JsonRpcServer:
        server = JsonRpcServer(token, **kwargs)
        await server.start()
        self.servers.append(server)
        return server

    async def connect(self, server: JsonRpcServer, token: str, **kwargs):
        client = await connect_json_rpc(
            "127.0.0.1", server.bound_port, token, **kwargs
        )
        self.clients.append(client)
        return client

    async def test_handshake_rejects_wrong_token_and_accepts_right_token(self) -> None:
        server = await self.make_server("correct-token")

        with self.assertRaises(AuthenticationError):
            await connect_json_rpc(
                "127.0.0.1", server.bound_port, "wrong-token"
            )
        self.assertEqual(server.peers, ())

        client = await self.connect(server, "correct-token")
        peer = await server.wait_for_peer(timeout=1)
        self.assertFalse(client.closed)
        self.assertFalse(peer.closed)

    async def test_requests_and_notifications_are_bidirectional(self) -> None:
        notice_received = asyncio.Event()

        async def add(params, _context):
            return params["left"] + params["right"]

        async def observe(params, _context):
            if params == {"message_id": 42}:
                notice_received.set()

        async def double(params, _context):
            return params["value"] * 2

        server = await self.make_server(
            "token", handlers={"math.add": add, "notice": observe}
        )
        client = await self.connect(
            server, "token", handlers={"math.double": double}
        )
        server_peer = await server.wait_for_peer(timeout=1)

        self.assertEqual(
            await client.request("math.add", {"left": 20, "right": 22}), 42
        )
        self.assertEqual(
            await server_peer.request("math.double", {"value": 21}), 42
        )
        await client.notify("notice", {"message_id": 42})
        await asyncio.wait_for(notice_received.wait(), 1)

    async def test_request_timeout_cancels_remote_handler_and_runs_hook(self) -> None:
        started = asyncio.Event()
        hook_called = asyncio.Event()
        handler_cancelled = asyncio.Event()

        async def hangs(_params, context):
            context.add_cancel_callback(hook_called.set)
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

        server = await self.make_server("token", handlers={"hang": hangs})
        client = await self.connect(server, "token")
        await server.wait_for_peer(timeout=1)

        with self.assertRaises(TimeoutError):
            await client.request("hang", {}, timeout=0.05)
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(hook_called.wait(), 1)
        await asyncio.wait_for(handler_cancelled.wait(), 1)

    async def test_duplicate_idempotency_key_executes_handler_once(self) -> None:
        executions = 0

        async def action(params, _context):
            nonlocal executions
            executions += 1
            await asyncio.sleep(0.03)
            return {"accepted": params["value"], "execution": executions}

        server = await self.make_server("token", handlers={"action": action})
        client = await self.connect(server, "token")
        await server.wait_for_peer(timeout=1)

        first, duplicate = await asyncio.gather(
            client.request("action", {"value": 7}, idempotency_key="same-action"),
            client.request("action", {"value": 7}, idempotency_key="same-action"),
        )
        replay = await client.request(
            "action", {"value": 7}, idempotency_key="same-action"
        )
        self.assertEqual(first, duplicate)
        self.assertEqual(first, replay)
        self.assertEqual(executions, 1)

        with self.assertRaises(JsonRpcError) as conflict:
            await client.request(
                "action", {"value": 8}, idempotency_key="same-action"
            )
        self.assertEqual(conflict.exception.code, -32009)
        self.assertEqual(executions, 1)

    async def test_idempotency_cache_is_shared_across_reconnections(self) -> None:
        executions = 0

        async def action(_params, _context):
            nonlocal executions
            executions += 1
            return executions

        server = await self.make_server("token", handlers={"action": action})
        first_client = await self.connect(server, "token")
        await server.wait_for_peer(timeout=1)
        self.assertEqual(
            await first_client.request("action", {}, idempotency_key="persistent"),
            1,
        )
        await first_client.close()

        second_client = await self.connect(server, "token")
        await server.wait_for_peer(timeout=1)
        self.assertEqual(
            await second_client.request("action", {}, idempotency_key="persistent"),
            1,
        )
        self.assertEqual(executions, 1)


class MediaPathValidatorTests(unittest.TestCase):
    def test_accepts_file_below_runtime_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "runtime"
            root.mkdir()
            media = root / "turn-1" / "photo.jpg"
            media.parent.mkdir()
            media.write_bytes(b"image")

            validator = MediaPathValidator(root)
            self.assertEqual(validator.validate("turn-1/photo.jpg"), media.resolve())
            self.assertEqual(validator.validate(media), media.resolve())

    def test_rejects_parent_traversal_absolute_escape_and_directory(self) -> None:
        with TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "runtime"
            root.mkdir()
            outside = base / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            child_directory = root / "turn-1"
            child_directory.mkdir()
            validator = MediaPathValidator(root)

            with self.assertRaises(MediaPathError):
                validator.validate("../secret.txt")
            with self.assertRaises(MediaPathError):
                validator.validate(outside)
            with self.assertRaises(MediaPathError):
                validator.validate(child_directory)
            with self.assertRaises(MediaPathError):
                validator.validate("../future.jpg", must_exist=False)


if __name__ == "__main__":
    unittest.main()
