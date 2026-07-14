"""Subprocess-only fake for MilanaService integration tests.

It speaks the production authenticated JSON-RPC protocol but never imports
Telethon and never accesses the network beyond the loopback service socket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from milana_ipc import RequestContext, connect_json_rpc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--dev-chat", action="store_true")
    return parser


class FakeHost:
    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir.resolve()
        self.seed = json.loads(
            (self.runtime_dir / "fake-notice.json").read_text(encoding="utf-8")
        )
        self.log_path = self.runtime_dir / "fake-actions.jsonl"
        with (self.runtime_dir / "fake-starts.txt").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("start\n")

    def _log(self, action: str, payload: Mapping[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"action": action, "payload": dict(payload)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    async def open(self, params: Any, _request: RequestContext) -> Mapping[str, Any]:
        payload = dict(params or {})
        target = payload.get("target_ref", self.seed["chat_id"])
        incoming = bool(payload.get("notice_ids"))
        return {
            "turn_id": payload["turn_id"],
            "target_token": "fake-target-token",
            "target_ref": target,
            "messages": (
                [
                    {
                        "message_id": self.seed["message_id"],
                        "timestamp": self.seed["timestamp"],
                        "sender": self.seed["sender"],
                        "text": self.seed["text"],
                        "media_type": self.seed["media_type"],
                    }
                ]
                if incoming
                else []
            ),
            "history": [],
        }

    async def execute(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        payload = dict(params or {})
        action = str(payload.get("action"))
        arguments = dict(payload.get("arguments") or {})
        self._log(
            action,
            {
                "arguments": arguments,
                "idempotency_key": request.idempotency_key,
            },
        )
        if action == "open_sticker_picker":
            if arguments.get("pack_id") is None:
                body = {"view": "index", "packs": [{"pack_id": "P001"}]}
            else:
                body = {
                    "view": "pack",
                    "stickers": [
                        {
                            "sticker_id": "P001:S001",
                            "set_id": 1,
                            "set_access_hash": 2,
                            "set_short_name": "fake",
                            "document_id": 3,
                            "pack_title": "Fake",
                            "emoji": "🙂",
                        }
                    ],
                }
            return {
                "status": "ok",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(body, ensure_ascii=False),
                    }
                ],
            }
        if action == "send_messages":
            return {"status": "sent", "sent_message_ids": [101]}
        if action == "send_sticker":
            return {"status": "sent", "message_id": 102}
        if action == "acknowledge":
            return {"status": "acknowledged"}
        return {"status": "ok"}

    async def cleanup(
        self, params: Any, _request: RequestContext
    ) -> Mapping[str, Any]:
        return {"cleaned": True, "turn_id": dict(params or {}).get("turn_id")}

    async def presence(
        self, params: Any, _request: RequestContext
    ) -> Mapping[str, Any]:
        return {"online": bool(dict(params or {}).get("online"))}


async def _main() -> None:
    args = _parser().parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    host = FakeHost(args.runtime_dir)
    peer = await connect_json_rpc(
        "127.0.0.1",
        args.port,
        token,
        handlers={
            "telegram.open": host.open,
            "telegram.execute": host.execute,
            "telegram.cleanup_turn": host.cleanup,
            "telegram.presence": host.presence,
        },
        request_timeout=10.0,
        name="fake-telegram-subprocess",
    )
    seed = host.seed
    if not (args.runtime_dir / "fake-disable-notice").exists():
        await peer.notify(
            "telegram.notice",
            {
                "source": "telegram",
                "notice_id": seed["notice_id"],
                "chat_id": seed["chat_id"],
                "message_id": seed["message_id"],
                "timestamp": seed["timestamp"],
                "sender": seed["sender"],
                "media_type": seed["media_type"],
            },
            idempotency_key=f"notice:{seed['notice_id']}",
        )
    crash_once = args.runtime_dir / "fake-crash-once"
    crash_marker = args.runtime_dir / "fake-crashed"
    if crash_once.exists() and not crash_marker.exists():
        crash_marker.write_text("crashed", encoding="utf-8")
        await peer.close()
        return
    await peer.wait_closed()


if __name__ == "__main__":
    asyncio.run(_main())
