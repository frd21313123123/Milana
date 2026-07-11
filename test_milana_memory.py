import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from milana_memory import (
    MAX_DIARY_ENTRY_LENGTH,
    MAX_MESSAGE_LENGTH,
    MilanaMemoryStore,
)


class MilanaMemoryStoreTests(unittest.TestCase):
    def test_history_is_persistent_ordered_limited_and_isolated(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store = MilanaMemoryStore(path)
            self.assertTrue(
                store.add_message(
                    100,
                    "user",
                    "Привет",
                    telegram_message_id=1,
                    sender_name="Анна",
                )
            )
            store.add_message(100, "assistant", "Привет!", telegram_message_id=1)
            store.add_message(100, "user", "Как дела?", telegram_message_id=2)
            store.add_message(200, "user", "Другой чат", telegram_message_id=1)
            self.assertFalse(
                store.add_message(100, "user", "Дубль", telegram_message_id=1)
            )
            store.close()

            reopened = MilanaMemoryStore(path)
            history = reopened.get_chat_history(100)
            self.assertEqual(
                [(item.role, item.content) for item in history],
                [
                    ("user", "Привет"),
                    ("assistant", "Привет!"),
                    ("user", "Как дела?"),
                ],
            )
            self.assertEqual(
                [item.content for item in reopened.get_chat_history(100, limit=2)],
                ["Привет!", "Как дела?"],
            )
            self.assertEqual(
                [item.content for item in reopened.get_chat_history(200)],
                ["Другой чат"],
            )
            reopened.close()

    def test_diary_is_global_persistent_and_deduplicated(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store = MilanaMemoryStore(path)
            self.assertTrue(
                store.add_diary_entry(
                    "Анна любит чай",
                    source_chat_id=100,
                    source_message_id=5,
                )
            )
            self.assertFalse(store.add_diary_entry("  анна любит чай  "))
            store.close()

            reopened = MilanaMemoryStore(path)
            entries = reopened.get_diary()
            self.assertEqual([entry.content for entry in entries], ["Анна любит чай"])
            self.assertEqual(entries[0].source_chat_id, "100")
            self.assertIn("Анна любит чай", reopened.diary_instructions())
            reopened.close()

    def test_invalid_values_are_rejected(self) -> None:
        store = MilanaMemoryStore()
        with self.assertRaises(ValueError):
            store.add_message(1, "system", "нет")
        with self.assertRaises(TypeError):
            store.add_message(1, "user", None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.add_message(1, "user", "x" * (MAX_MESSAGE_LENGTH + 1))
        with self.assertRaises(ValueError):
            store.add_diary_entry("   ")
        with self.assertRaises(ValueError):
            store.add_diary_entry("x" * (MAX_DIARY_ENTRY_LENGTH + 1))
        store.close()

    def test_empty_store_and_non_positive_limits(self) -> None:
        store = MilanaMemoryStore()

        self.assertFalse(store.has_chat_history("missing"))
        self.assertIsNone(store.latest_telegram_message_id("missing"))
        self.assertEqual(store.get_chat_history("missing", limit=0), [])
        self.assertEqual(store.get_diary(limit=-1), [])
        self.assertIn("Дневник пока пуст", store.diary_instructions())

        store.close()

    def test_latest_message_id_ignores_local_turns_without_telegram_id(self) -> None:
        store = MilanaMemoryStore()
        store.add_message(1, "assistant", "Локальный ответ")
        store.add_message(1, "user", "Первое", telegram_message_id=7)
        store.add_message(1, "user", "Второе", telegram_message_id=11)

        self.assertTrue(store.has_chat_history(1))
        self.assertEqual(store.latest_telegram_message_id(1), 11)

        store.close()

    def test_sender_and_explicit_timestamps_are_normalized_and_preserved(self) -> None:
        store = MilanaMemoryStore()
        store.add_message(
            1,
            "user",
            "  Привет  ",
            sender_name="  Анна  ",
            created_at="2026-07-11T10:00:00+00:00",
        )

        message = store.get_chat_history(1)[0]
        self.assertEqual(message.content, "Привет")
        self.assertEqual(message.sender_name, "Анна")
        self.assertEqual(message.created_at, "2026-07-11T10:00:00+00:00")

        store.close()

    def test_response_input_uses_only_requested_chat(self) -> None:
        store = MilanaMemoryStore()
        store.add_message(1, "user", "Первый", sender_name="Ира")
        store.add_message(1, "assistant", "Ответ")
        store.add_message(2, "user", "Секрет другого чата")

        self.assertEqual(
            store.response_input(1),
            [
                {"role": "user", "content": "Ира: Первый"},
                {"role": "assistant", "content": "Ответ"},
            ],
        )
        store.close()

    def test_chat_summary_persists_and_is_isolated(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            store = MilanaMemoryStore(path)
            store.set_chat_summary(100, "Анна любит кофе и планирует поездку.", covered_user_messages=42, last_covered_message_id=99)
            store.set_chat_summary(200, "Другой пользователь — только работа.")

            info100 = store.get_chat_summary_info(100)
            info200 = store.get_chat_summary_info(200)
            self.assertIsNotNone(info100)
            self.assertIn("кофе", info100.summary if info100 else "")
            self.assertEqual(info100.covered_user_messages if info100 else 0, 42)
            self.assertIn("работа", info200.summary if info200 else "")

            # response_input_with_summary injects the block
            store.add_message(100, "user", "Привет снова", sender_name="Анна")
            inp = store.response_input_with_summary(100, recent_limit=5)
            self.assertTrue(any("Краткий обзор предыдущей части" in (m.get("content") or "") for m in inp))
            self.assertTrue(any("Привет снова" in (m.get("content") or "") for m in inp))
            store.close()

    def test_user_window_counts_and_nth_last_queries(self) -> None:
        store = MilanaMemoryStore()
        for i in range(65):
            store.add_message(5, "user", f"msg {i}")
            if i % 3 == 0:
                store.add_message(5, "assistant", "ok")

        self.assertEqual(store.count_user_messages(5), 65)
        # 30th last user id exists
        uid = store.get_nth_last_user_message_id(5, 30)
        self.assertIsNotNone(uid)
        self.assertIsInstance(uid, int)

        total_last = store.get_nth_last_message_id(5, 30)
        self.assertIsNotNone(total_last)

        # range helper
        batch = store.get_messages_in_id_range(5, 1, 10)
        self.assertTrue(len(batch) >= 1)
        store.close()

    def test_response_input_with_summary_uses_30_and_skips_summary_when_absent(self) -> None:
        store = MilanaMemoryStore()
        for i in range(40):
            store.add_message(9, "user", f"q{i}")
        inp = store.response_input_with_summary(9, recent_limit=30)
        # No summary yet -> starts directly with messages, length == 30
        self.assertEqual(len(inp), 30)
        self.assertNotIn("Краткий обзор", str(inp))
        store.close()

    def test_response_input_excludes_active_users_before_applying_recent_limit(self) -> None:
        store = MilanaMemoryStore()
        for message_id in range(1, 33):
            store.add_message(
                9,
                "user",
                f"q{message_id}",
                telegram_message_id=message_id,
                sender_name="Лена",
            )

        inp = store.response_input_with_summary(
            9,
            recent_limit=30,
            exclude_user_message_ids={31, 32},
        )

        self.assertEqual(len(inp), 30)
        self.assertIn("q1", inp[0]["content"])
        self.assertIn("q30", inp[-1]["content"])
        self.assertNotIn("q31", str(inp))
        self.assertNotIn("q32", str(inp))
        store.close()

    def test_active_user_exclusion_keeps_summary_and_assistant_with_same_id(self) -> None:
        store = MilanaMemoryStore()
        store.set_chat_summary(7, "Старый контекст")
        store.add_message(7, "user", "Новый вопрос", telegram_message_id=10)
        store.add_message(7, "assistant", "Уже отправлено", telegram_message_id=10)

        inp = store.response_input_with_summary(
            7,
            exclude_user_message_ids={10},
        )

        self.assertIn("Старый контекст", inp[0]["content"])
        self.assertNotIn("Новый вопрос", str(inp))
        self.assertIn({"role": "assistant", "content": "Уже отправлено"}, inp)
        store.close()


if __name__ == "__main__":
    unittest.main()
