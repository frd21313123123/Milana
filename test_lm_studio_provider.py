import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import BadRequestError

from lm_studio_provider import LMStudioModelClient


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def request(*, tools=()):
    return {
        "model": "local-model",
        "instructions": "Ответь по схеме.",
        "input": [{"role": "user", "content": "привет"}],
        "tools": list(tools),
        "max_output_tokens": 100,
        "temperature": 0.2,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "test_schema",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    }


def chat_result(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def grammar_error():
    request = httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        "Failed to initialize samplers: failed to parse grammar",
        response=response,
        body={"error": "failed to parse grammar"},
    )


class LMStudioProviderTests(unittest.IsolatedAsyncioTestCase):
    def client(self):
        raw = MagicMock()
        raw.responses.create = AsyncMock()
        raw.chat.completions.create = AsyncMock()
        return LMStudioModelClient(raw), raw

    async def test_toolless_structured_request_uses_chat_schema_directly(self):
        client, raw = self.client()
        raw.chat.completions.create.return_value = chat_result(
            json.dumps({"answer": "готово"}, ensure_ascii=False)
        )

        response = await client.responses.create(**request())

        self.assertEqual(json.loads(response.output_text), {"answer": "готово"})
        raw.responses.create.assert_not_awaited()
        kwargs = raw.chat.completions.create.await_args.kwargs
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertEqual(kwargs["response_format"]["json_schema"]["schema"], SCHEMA)

    async def test_responses_tool_call_is_not_reformatted(self):
        client, raw = self.client()
        tool_response = SimpleNamespace(
            output=[SimpleNamespace(type="function_call")], output_text=""
        )
        raw.responses.create.return_value = tool_response

        response = await client.responses.create(
            **request(tools=[{"type": "function", "name": "inspect"}])
        )

        self.assertIs(response, tool_response)
        raw.chat.completions.create.assert_not_awaited()

    async def test_plain_responses_final_is_repaired_with_chat_schema(self):
        client, raw = self.client()
        raw.responses.create.return_value = SimpleNamespace(
            output=[], output_text="обычный текст"
        )
        raw.chat.completions.create.return_value = chat_result('{"answer":"готово"}')

        response = await client.responses.create(
            **request(tools=[{"type": "function", "name": "inspect"}])
        )

        self.assertEqual(json.loads(response.output_text), {"answer": "готово"})
        repair_messages = raw.chat.completions.create.await_args.kwargs["messages"]
        self.assertIn("обычный текст", repair_messages[1]["content"])

    async def test_request_without_schema_stays_on_responses_api(self):
        client, raw = self.client()
        expected = SimpleNamespace(output=[], output_text="обычный ответ")
        raw.responses.create.return_value = expected
        plain = request()
        plain.pop("text")

        response = await client.responses.create(**plain)

        self.assertIs(response, expected)
        raw.chat.completions.create.assert_not_awaited()

    async def test_grammar_failure_falls_back_to_schema_prompt_and_is_remembered(self):
        client, raw = self.client()
        repaired = chat_result('{"answer":"готово"}')
        raw.chat.completions.create.side_effect = [
            grammar_error(),
            repaired,
            repaired,
        ]

        first = await client.responses.create(**request())
        second = await client.responses.create(**request())

        self.assertEqual(json.loads(first.output_text), {"answer": "готово"})
        self.assertEqual(json.loads(second.output_text), {"answer": "готово"})
        formats = [
            call.kwargs["response_format"]["type"]
            for call in raw.chat.completions.create.await_args_list
        ]
        self.assertEqual(formats, ["json_schema", "text", "text"])

    async def test_invalid_nested_arguments_json_is_discarded_safely(self):
        client, raw = self.client()
        raw.chat.completions.create.return_value = chat_result(
            json.dumps(
                {
                    "answer": "готово",
                    "entity_updates": [
                        {"arguments_json": "не json"},
                        {"arguments_json": '{"id":"friend"}'},
                    ],
                },
                ensure_ascii=False,
            )
        )

        response = await client.responses.create(**request())
        payload = json.loads(response.output_text)

        self.assertEqual(
            payload["entity_updates"],
            [{"arguments_json": '{"id":"friend"}'}],
        )


if __name__ == "__main__":
    unittest.main()
