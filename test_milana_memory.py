import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from milana_memory import MAX_DIARY_ENTRY_LENGTH, MilanaMemoryStore


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
        with self.assertRaises(ValueError):
            store.add_diary_entry("   ")
        with self.assertRaises(ValueError):
            store.add_diary_entry("x" * (MAX_DIARY_ENTRY_LENGTH + 1))
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


if __name__ == "__main__":
    unittest.main()
