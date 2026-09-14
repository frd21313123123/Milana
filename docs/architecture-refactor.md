# Architecture refactor roadmap

Крупные runtime-файлы стоит делить без изменения публичных точек входа.

Предлагаемый порядок:

1. `telegram_client.py`: вынести media, outbox, presence и incoming routing.
2. `milana_service.py`: вынести bootstrap, Telegram turn orchestration и heartbeat turns.
3. `milana_state.py`: разделить entities, relationships, goals, events и repository.
4. После каждого шага сохранять compatibility imports и запускать полный CI.

Главный критерий такого рефакторинга: никаких изменений семантики durable outbox,
idempotency keys, acknowledgement и recovery в том же коммите, где перемещается код.
