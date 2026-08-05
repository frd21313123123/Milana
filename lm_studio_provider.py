"""Responses-compatible LM Studio client with enforced JSON Schema output."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from openai import BadRequestError


class _StructuredResponse:
    """Expose Chat Completions text through the Responses fields Milana uses."""

    def __init__(self, source: Any, output_text: str) -> None:
        self._source = source
        self.output_text = output_text
        self.output: list[Any] = []
        self.status = "completed"
        self.incomplete_details = None
        self.metadata: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class _LMStudioResponses:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._supports_schema_grammar: bool | None = None

    async def create(self, **request: Any) -> Any:
        format_spec = self._format_spec(request)
        if format_spec is not None and not request.get("tools"):
            try:
                return await self._structured_chat(request, format_spec)
            except BadRequestError:
                # Older LM Studio builds may not accept structured Chat
                # Completions. Keep the Responses endpoint as a compatibility
                # fallback and repair its final draft below.
                pass

        response = await self._client.responses.create(**request)
        if format_spec is None or self._has_tool_calls(response):
            return response
        return await self._structured_chat(
            request,
            format_spec,
            draft=str(getattr(response, "output_text", "") or ""),
        )

    async def _structured_chat(
        self,
        request: Mapping[str, Any],
        format_spec: Mapping[str, Any],
        *,
        draft: str | None = None,
    ) -> _StructuredResponse:
        if draft is None:
            messages = self._direct_messages(request)
        else:
            messages = self._repair_messages(request, draft)

        schema = format_spec["schema"]
        options = {
            "model": request["model"],
            "messages": messages,
            "temperature": float(request.get("temperature", 0.0)),
            "max_tokens": max(512, int(request.get("max_output_tokens", 512))),
        }
        completion: Any
        if self._supports_schema_grammar is not False:
            try:
                completion = await self._client.chat.completions.create(
                    **options,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": str(
                                format_spec.get("name") or "milana_agent_turn"
                            ),
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                self._supports_schema_grammar = True
            except BadRequestError as exc:
                if not self._grammar_unsupported(exc):
                    raise
                self._supports_schema_grammar = False
                completion = await self._schema_prompt_completion(options, schema)
        else:
            completion = await self._schema_prompt_completion(options, schema)
        choices = list(getattr(completion, "choices", None) or [])
        content = (
            str(getattr(getattr(choices[0], "message", None), "content", "") or "")
            if choices
            else ""
        ).strip()
        if not content:
            raise ValueError("LM Studio вернула пустой Structured Output")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LM Studio нарушила JSON Schema в Chat Completions"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("LM Studio вернула не JSON-объект")
        payload = self._sanitize_arguments_json(payload)
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return _StructuredResponse(completion, canonical)

    async def _schema_prompt_completion(
        self, options: Mapping[str, Any], schema: Mapping[str, Any]
    ) -> Any:
        messages = list(options["messages"])
        schema_text = (
            "Обязательный контракт финального ответа. Верни только JSON-объект, "
            "который точно соответствует этой JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                **messages[0],
                "content": str(messages[0].get("content", "")) + "\n\n" + schema_text,
            }
        else:
            messages.insert(0, {"role": "system", "content": schema_text})
        return await self._client.chat.completions.create(
            **{**dict(options), "messages": messages},
            response_format={"type": "text"},
        )

    @classmethod
    def _direct_messages(cls, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        instructions = str(request.get("instructions", "") or "").strip()
        if instructions:
            messages.append({"role": "system", "content": instructions})
        raw_input = request.get("input", [])
        items = raw_input if isinstance(raw_input, list) else [raw_input]
        for item in items:
            converted = cls._chat_message(item)
            if converted is not None:
                messages.append(converted)
        if not messages:
            messages.append({"role": "user", "content": "Сформируй финальный JSON."})
        return messages

    @classmethod
    def _repair_messages(
        cls, request: Mapping[str, Any], draft: str
    ) -> list[dict[str, Any]]:
        context = {
            "instructions": request.get("instructions", ""),
            "input": cls._json_ready(request.get("input", [])),
            "draft": draft,
        }
        return [
            {
                "role": "system",
            "content": (
                "Ты транспортный JSON-форматтер. Верни только объект по "
                "заданной JSON Schema. Сохрани намерение draft, не добавляй "
                "новых внешних действий и точно копируй служебные токены из "
                "input. Если draft пуст, сформируй минимальный безопасный "
                "финальный объект из instructions и input. Все поля "
                "arguments_json должны быть строками, содержащими валидный "
                "JSON-объект; если точного изменения нет, верни пустой массив."
            ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    @classmethod
    def _chat_message(cls, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        role = str(item.get("role", "user") or "user")
        content = item.get("content", "")
        if isinstance(content, str):
            return {"role": role, "content": content}
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            return {"role": role, "content": str(content)}
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "input_text":
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            elif kind == "input_image" and part.get("image_url"):
                image_url: dict[str, Any] = {"url": part["image_url"]}
                if part.get("detail"):
                    image_url["detail"] = part["detail"]
                parts.append({"type": "image_url", "image_url": image_url})
        return {"role": role, "content": parts} if parts else None

    @staticmethod
    def _format_spec(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        text = request.get("text")
        if not isinstance(text, Mapping):
            return None
        format_spec = text.get("format")
        if (
            not isinstance(format_spec, Mapping)
            or format_spec.get("type") != "json_schema"
            or not isinstance(format_spec.get("schema"), Mapping)
        ):
            return None
        return format_spec

    @staticmethod
    def _has_tool_calls(response: Any) -> bool:
        if getattr(response, "agy_tool_calls", None):
            return True
        return any(
            getattr(item, "type", None) == "function_call"
            for item in list(getattr(response, "output", None) or [])
        )

    @staticmethod
    def _grammar_unsupported(exc: BadRequestError) -> bool:
        message = str(exc).lower()
        return (
            "failed to parse grammar" in message
            or "failed to initialize samplers" in message
        )

    @staticmethod
    def _sanitize_arguments_json(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        for key in (
            "entity_updates",
            "life_events",
            "goal_updates",
            "relationship_updates",
        ):
            items = result.get(key)
            if not isinstance(items, list):
                continue
            valid: list[dict[str, str]] = []
            for item in items:
                if not isinstance(item, Mapping) or set(item) != {"arguments_json"}:
                    continue
                raw = item.get("arguments_json")
                if not isinstance(raw, str):
                    continue
                try:
                    arguments = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(arguments, dict):
                    valid.append({
                        "arguments_json": json.dumps(
                            arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    })
            result[key] = valid
        return result

    @classmethod
    def _json_ready(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._json_ready(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls._json_ready(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return cls._json_ready(model_dump())
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


class LMStudioModelClient:
    """Use LM Studio tools via Responses and enforce finals via JSON Schema."""

    def __init__(self, openai_client: Any) -> None:
        self.openai_client = openai_client
        self.responses = _LMStudioResponses(openai_client)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.openai_client, name)
