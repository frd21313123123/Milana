import unittest
import os
from unittest.mock import patch

import httpx

from jev_provider import (
    HeartbeatRouteDecision,
    JevConfig,
    JevDecisionClient,
    JevLowConfidenceError,
    JevResult,
    JevUnavailableError,
    JevValidationError,
    OpenLoopCandidate,
    OpenLoopSnapshot,
)
from telegram_client import load_jev_config


def _payload(*, confidence=0.92):
    return {
        "model": "jev-1.13.0",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "reflect",
                "confidence": confidence,
                "probabilities": {"reflect": confidence, "skip": 1 - confidence},
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class JevResultTests(unittest.TestCase):
    def test_missing_key_disables_jev_without_rejecting_config(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_jev_config({"jev": {"enabled": True}}, {})
        self.assertFalse(config.enabled)
        self.assertEqual(config.api_key, "")

    def test_choice_is_strict_and_confidence_gated(self):
        result = JevResult(
            model="jev-1.13.0",
            answers=_payload()["answers"],
            input_tokens=100,
            output_tokens=20,
            elapsed_ms=10,
        )
        self.assertEqual(
            result.choice(
                "route", allowed={"reflect", "skip"}, minimum_confidence=0.8
            ).choice,
            "reflect",
        )
        with self.assertRaises(JevLowConfidenceError):
            result.choice(
                "route", allowed={"reflect", "skip"}, minimum_confidence=0.95
            )
        with self.assertRaises(JevValidationError):
            result.choice("route", allowed={"skip"}, minimum_confidence=0)

    def test_open_loop_snapshot_exposes_only_actionable_items(self):
        snapshot = OpenLoopSnapshot(
            (
                OpenLoopCandidate("goal:1", "goal", "закончить проект"),
                OpenLoopCandidate(
                    "relationship:1",
                    "awaiting_reply",
                    "ждать ответа",
                    actionable=False,
                ),
            )
        )
        self.assertEqual([item.id for item in snapshot.actionable], ["goal:1"])
        self.assertIsNone(HeartbeatRouteDecision("reflect").target_ref)


class JevDecisionClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_request_is_validated(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/v1/systemone")
            self.assertEqual(request.headers["authorization"], "Bearer secret")
            return httpx.Response(200, json=_payload())

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(
            transport=transport, base_url="https://api.typesafe.ai"
        )
        client = JevDecisionClient(
            JevConfig(enabled=True, api_key="secret"), client=http
        )
        try:
            result = await client.evaluate(
                scenario="test",
                state={"value": 1},
                questions={
                    "route": {
                        "type": "choice",
                        "instructions": "Choose",
                        "criteria": {"reflect": "Reflect", "skip": "Skip"},
                    }
                },
            )
        finally:
            await http.aclose()
        self.assertEqual(result.model, "jev-1.13.0")
        self.assertEqual(result.input_tokens, 100)

    async def test_three_failures_open_circuit_and_probe_recovers(self):
        clock = [100.0]
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            if calls <= 3:
                return httpx.Response(503, json={"error": "offline"})
            return httpx.Response(200, json=_payload())

        transport = httpx.MockTransport(handler)
        http = httpx.AsyncClient(
            transport=transport, base_url="https://api.typesafe.ai"
        )
        client = JevDecisionClient(
            JevConfig(
                enabled=True,
                api_key="secret",
                failure_threshold=3,
                cooldown_seconds=300,
            ),
            client=http,
            monotonic=lambda: clock[0],
        )
        request = {
            "scenario": "test",
            "state": {},
            "questions": {"ok": {"type": "noul", "instructions": "Is it ok?"}},
        }
        try:
            for _ in range(3):
                with self.assertRaises(JevUnavailableError):
                    await client.evaluate(**request)
            self.assertTrue(client.circuit_open)
            with self.assertRaises(JevUnavailableError):
                await client.evaluate(**request)
            self.assertEqual(calls, 3)
            clock[0] += 301
            result = await client.evaluate(**request)
            self.assertEqual(result.model, "jev-1.13.0")
            self.assertFalse(client.circuit_open)
        finally:
            await http.aclose()

    async def test_malformed_json_uses_unavailable_path(self):
        async def handler(_request):
            return httpx.Response(200, text="not-json")

        http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.typesafe.ai",
        )
        client = JevDecisionClient(
            JevConfig(enabled=True, api_key="secret"), client=http
        )
        try:
            with self.assertRaises(JevValidationError):
                await client.evaluate(
                    scenario="test",
                    state={},
                    questions={
                        "ok": {"type": "noul", "instructions": "Is it ok?"}
                    },
                )
        finally:
            await http.aclose()

    async def test_auth_rate_limit_server_and_timeout_fail_without_retry(self):
        failures = (401, 403, 429, 503, "timeout")
        for failure in failures:
            with self.subTest(failure=failure):
                calls = 0

                async def handler(request):
                    nonlocal calls
                    calls += 1
                    if failure == "timeout":
                        raise httpx.ReadTimeout("slow Jev", request=request)
                    return httpx.Response(failure, json={"error": "unavailable"})

                http = httpx.AsyncClient(
                    transport=httpx.MockTransport(handler),
                    base_url="https://api.typesafe.ai",
                )
                client = JevDecisionClient(
                    JevConfig(enabled=True, api_key="secret"), client=http
                )
                try:
                    with self.assertRaises(JevUnavailableError):
                        await client.evaluate(
                            scenario="test",
                            state={},
                            questions={
                                "ok": {
                                    "type": "noul",
                                    "instructions": "Is it ok?",
                                }
                            },
                        )
                finally:
                    await http.aclose()
                self.assertEqual(calls, 1)

    async def test_invalid_response_schema_is_rejected(self):
        async def handler(_request):
            return httpx.Response(
                200,
                json={"model": "jev-1.13.0", "answers": {}, "usage": {}},
            )

        http = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.typesafe.ai",
        )
        client = JevDecisionClient(
            JevConfig(enabled=True, api_key="secret"), client=http
        )
        try:
            with self.assertRaises(JevValidationError):
                await client.evaluate(
                    scenario="test",
                    state={},
                    questions={
                        "ok": {"type": "noul", "instructions": "Is it ok?"}
                    },
                )
        finally:
            await http.aclose()


if __name__ == "__main__":
    unittest.main()
