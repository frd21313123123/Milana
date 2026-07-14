"""Measure Milana's Telegram fast path without contacting Telegram.

The benchmark deliberately uses the production ``MilanaService`` and model
client, but replaces the Telegram skill host with an in-process deterministic
supervisor.  Consequently, running this module can invoke Gemini CLI, yet can
never send a message to a real Telegram account.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Mapping

from agy_provider import AgyModelClient
from milana import TurnTrigger
from milana_memory import MilanaMemoryStore
from milana_schedule import load_routine
from milana_service import MilanaService
from milana_state import MilanaStateStore
from telegram_client import (
    AIConfig,
    GEMINI_LLM_CHOICE,
    MessageFlowConfig,
    load_ai_config,
)


DEFAULT_RUNS = 50
BENCHMARK_CHAT_ID = 7_700_000_001
BENCHMARK_TARGET_TOKEN = "benchmark-target-token"


class FakeTelegramSupervisor:
    """Minimal Telegram host contract backed only by local memory."""

    def __init__(self, *, chat_id: int = BENCHMARK_CHAT_ID) -> None:
        self.chat_id = chat_id
        self._notices: dict[str, dict[str, Any]] = {}
        self.send_calls = 0
        self.sent_messages = 0
        self.sent_message_ids: list[int] = []

    def register_notice(self, payload: Mapping[str, Any], *, text: str) -> None:
        notice_id = payload.get("notice_id")
        if not isinstance(notice_id, str) or not notice_id:
            raise ValueError("Benchmark notice_id must be a non-empty string")
        materialized = dict(payload)
        materialized["text"] = text
        self._notices[notice_id] = materialized

    async def request(
        self, method: str, params: Mapping[str, Any], **_options: Any
    ) -> Mapping[str, Any]:
        if method == "telegram.open":
            notice_ids = params.get("notice_ids", [])
            if not isinstance(notice_ids, list):
                raise TypeError("telegram.open notice_ids must be an array")
            messages = []
            for notice_id in notice_ids:
                if not isinstance(notice_id, str) or notice_id not in self._notices:
                    raise LookupError(f"Unknown benchmark notice: {notice_id!r}")
                notice = self._notices[notice_id]
                messages.append(
                    {
                        "message_id": notice["message_id"],
                        "timestamp": notice["timestamp"],
                        "sender": notice["sender"],
                        "text": notice["text"],
                        "media_type": "text",
                    }
                )
            return {
                "turn_id": params["turn_id"],
                "target_token": BENCHMARK_TARGET_TOKEN,
                "target_ref": self.chat_id,
                "messages": messages,
                "history": [],
            }

        if method == "telegram.execute":
            action = params.get("action")
            if action == "send_messages":
                arguments = params.get("arguments", {})
                if not isinstance(arguments, Mapping):
                    raise TypeError("send_messages arguments must be an object")
                messages = arguments.get("messages", [])
                start_index = arguments.get("start_index", 0)
                if not isinstance(messages, list) or not isinstance(start_index, int):
                    raise TypeError("Invalid send_messages benchmark payload")
                indexes = list(range(start_index, len(messages)))
                self.send_calls += 1
                self.sent_messages += len(indexes)
                base = 1_000_000 + self.sent_messages - len(indexes)
                receipts = [base + offset for offset in range(len(indexes))]
                self.sent_message_ids.extend(receipts)
                return {
                    "status": "sent",
                    "sent_message_ids": receipts,
                    "sent_part_indexes": indexes,
                    "deduplicated_part_indexes": [],
                    "next_part_index": len(messages),
                    "total_parts": len(messages),
                    "first_send_elapsed_ms": 0.0 if indexes else None,
                }
            # typing and acknowledge are local no-ops in this benchmark.
            return {"status": "ok"}

        if method == "telegram.cleanup_turn":
            return {"cleaned": True}
        if method == "telegram.presence":
            return {"online": bool(params.get("online"))}
        raise LookupError(f"Unsupported fake Telegram method: {method}")

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def status(self) -> Mapping[str, Any]:
        return {
            "connected": True,
            "kind": "local-benchmark-host",
            "send_calls": self.send_calls,
            "sent_messages": self.sent_messages,
            "receipts": len(self.sent_message_ids),
        }


def benchmark_config(base: AIConfig) -> AIConfig:
    """Force the production limits used by the dev-chat fast-path rollout."""

    return replace(
        base,
        message_flow=MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            max_reply_messages=1,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        ),
        telegram_fast_path=replace(
            base.telegram_fast_path,
            enabled=True,
            dev_chat_only=True,
            max_reply_messages=1,
        ),
    )


async def run_benchmark(
    *,
    runs: int = DEFAULT_RUNS,
    config: AIConfig | None = None,
    model_client: Any | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run sequential ordinary-text turns and return a JSON-safe report."""

    if isinstance(runs, bool) or not isinstance(runs, int) or runs <= 0:
        raise ValueError("runs must be a positive integer")
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    loaded_config = config or load_ai_config()
    if loaded_config.provider != GEMINI_LLM_CHOICE and model_client is None:
        raise ValueError(
            "This benchmark requires Gemini in llm_choice.txt; "
            "it never falls back to another provider"
        )
    effective_config = benchmark_config(loaded_config)
    effective_model = model_client or AgyModelClient(
        model=effective_config.model,
        timeout_seconds=int(timeout_seconds),
    )

    failures: list[dict[str, Any]] = []
    completed_turns = 0
    successful_first_sends = 0
    known_model_calls = 0
    supervisor = FakeTelegramSupervisor()

    with TemporaryDirectory(prefix="milana-fast-benchmark-") as raw_directory:
        database_path = Path(raw_directory) / "benchmark.sqlite3"
        memory = MilanaMemoryStore(database_path)
        state = MilanaStateStore(database_path)
        service = MilanaService(
            config=effective_config,
            model_client=effective_model,
            memory=memory,
            state=state,
            routine=load_routine(),
            rpc_server=SimpleNamespace(),
            supervisor=supervisor,
            dev_mode=True,
            now=lambda: datetime.now(timezone.utc),
        )
        # Keep the acceptance sample at exactly ``runs`` real CLI calls. The
        # independently tested post-send compactor is deliberately excluded
        # from this foreground latency benchmark.
        service._schedule_summary_compaction = lambda _chat_id: None  # type: ignore[method-assign]
        try:
            for index in range(1, runs + 1):
                occurred_at = datetime.now(timezone.utc)
                notice_id = f"tg:{supervisor.chat_id}:{index}"
                notice = {
                    "source": "telegram",
                    "notice_id": notice_id,
                    "chat_id": supervisor.chat_id,
                    "message_id": index,
                    "timestamp": occurred_at.isoformat(),
                    "sender": {"id": 88, "display_name": "Benchmark User"},
                    "media_type": "text",
                }
                supervisor.register_notice(
                    notice,
                    text=f"Тестовое сообщение номер {index}. Ответь коротко.",
                )
                state.record_telegram_notice(notice, received_at=occurred_at)
                trigger = TurnTrigger(
                    kind="telegram_notice",
                    occurred_at=occurred_at,
                    source_skill="telegram",
                    revision=state.get_agent_state().revision,
                    metadata={
                        "chat_id": supervisor.chat_id,
                        "notice_ids": [notice_id],
                        "notices": [notice],
                    },
                )
                try:
                    result = await service._execute_turn(trigger)
                    completed_turns += 1
                    known_model_calls += int(result.model_rounds)
                    telegram = result.payload.get("telegram")
                    messages = (
                        telegram.get("messages", [])
                        if isinstance(telegram, Mapping)
                        else []
                    )
                    if isinstance(messages, list) and any(
                        isinstance(message, str) and message.strip()
                        for message in messages
                    ):
                        successful_first_sends += 1
                    else:
                        failures.append(
                            {
                                "run": index,
                                "error": "no_content_message",
                            }
                        )
                except Exception as exc:  # keep the 50-run sample auditable
                    failures.append(
                        {
                            "run": index,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

            # Let immediate best-effort typing tasks finish before closing the
            # temporary stores. They never contribute to first-send latency.
            pending_cosmetic = tuple(service.telegram_executor._typing_tasks)
            if pending_cosmetic:
                await asyncio.gather(*pending_cosmetic, return_exceptions=True)
            pending_presence = tuple(service._cosmetic_tasks)
            if pending_presence:
                await asyncio.gather(*pending_presence, return_exceptions=True)

            latency = state.telegram_latency_summary(
                limit=max(runs, 1),
                target_seconds=effective_config.telegram_fast_path.target_first_send_seconds,
            )
        finally:
            background_tasks = (
                *tuple(service.telegram_executor._typing_tasks),
                *tuple(service._cosmetic_tasks),
                *tuple(service._summary_tasks.values()),
            )
            for task in background_tasks:
                task.cancel()
            await asyncio.gather(
                *background_tasks,
                return_exceptions=True,
            )
            state.close()
            memory.close()

    requested_model_call_mean = known_model_calls / runs
    p95_ms = latency.get("p95_ms")
    target_ms = float(latency["target_ms"])
    receipt_count = len(supervisor.sent_message_ids)
    measured_samples = int(latency.get("sample_size", 0) or 0)
    sla_met = (
        successful_first_sends == runs
        and receipt_count == runs
        and measured_samples == runs
        and isinstance(p95_ms, (int, float))
        and float(p95_ms) <= target_ms
        and abs(requested_model_call_mean - 1.0) < 1e-9
    )
    return {
        "runs": runs,
        "success": successful_first_sends,
        "receipts": receipt_count,
        "completed_turns": completed_turns,
        "failures": len(failures),
        "failure_details": failures,
        "model_calls": {
            "known_total": known_model_calls,
            "mean": round(requested_model_call_mean, 6),
            "mean_per_completed_turn": (
                round(known_model_calls / completed_turns, 6)
                if completed_turns
                else None
            ),
        },
        "generation_to_first_send_ms": latency["generation_to_first_send_ms"],
        "phase_metrics_ms": latency["phases"],
        "sla": {
            "target_ms": target_ms,
            "sample_size": latency["sample_size"],
            "p95_ms": p95_ms,
            "exceedances": latency["exceedances"],
            "exceed_rate": latency["exceed_rate"],
            "met": sla_met,
            "checks": {
                "success_count": successful_first_sends == runs,
                "receipt_count": receipt_count == runs,
                "sample_count": measured_samples == runs,
                "mean_model_calls": abs(requested_model_call_mean - 1.0) < 1e-9,
                "p95": (
                    isinstance(p95_ms, (int, float))
                    and float(p95_ms) <= target_ms
                ),
            },
        },
        "outcomes": latency["outcomes"],
        "fake_telegram_host": dict(supervisor.status()),
    }


def _positive_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real Gemini CLI generations through MilanaService while using "
            "a local fake Telegram host"
        )
    )
    parser.add_argument("--runs", type=_positive_integer, default=DEFAULT_RUNS)
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_integer,
        default=300,
        help="total Gemini CLI deadline for one turn",
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows services/terminals can inherit cp1252 even though benchmark
    # diagnostics and model errors are Russian. Keep the JSON report lossless.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(
            run_benchmark(runs=args.runs, timeout_seconds=args.timeout_seconds)
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0 if result["sla"]["met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
