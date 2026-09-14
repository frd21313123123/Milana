# Architecture refactor

Крупные runtime-модули разделены без изменения публичных точек входа и runtime-семантики.

## Telegram client

`telegram_client.py` остаётся compatibility entrypoint, а самостоятельные обязанности вынесены в:

- `milana/telegram_config.py` - конфигурация, dataclass-модели и загрузка настроек;
- `milana/telegram_cli.py` - CLI parsing и presentation helpers;
- `milana/telegram_media.py` - MIME detection, загрузка media, GIF conversion и sticker rendering.

Legacy imports и monkey-patch paths, используемые тестами, сохранены.

## Milana service

Provider-neutral валидация state/heartbeat payload вынесена из `milana_service.py` в
`milana/service_state.py`. Service остаётся владельцем orchestration, model loop и skill lifecycle.

## Persistent state

`milana_state.py` теперь является facade/composition root для SQLite state store. Реализация разделена на:

- `milana/state_models.py` - dataclass-модели, ошибки, validation primitives и initiative policy;
- `milana/state_schema.py` - SQLite schema bootstrap, additive migrations и compatibility backfills;
- `milana/state_telegram.py` - Telegram notice journal, durable outbox, ack intents и latency metrics;
- `milana/state_lifecycle.py` - agent state, needs, recovery windows и heartbeat jobs;
- `milana/state_world.py` - entities, facts, life events, goals, relationships, summaries и atomic world updates.

`MilanaStateStore` собирает эти части через mixins и сохраняет прежний публичный API и re-export names.

## Safety invariant

Перемещение persistence-кода не меняет durable outbox semantics, idempotency keys,
acknowledgement, additive migrations или recovery behavior. Каждый этап принимался только после
успешных `compileall`, Ruff и полного набора unit/integration tests.
