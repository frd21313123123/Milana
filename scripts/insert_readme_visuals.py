from pathlib import Path

README = Path("README.md")
START = "<!-- milana-showcase:start -->"
END = "<!-- milana-showcase:end -->"

SHOWCASE = f"""{START}

<p align="center">
  <img src="docs/assets/hero.webp" alt="Milana AI companion" width="100%">
</p>

<p align="center">
  <b>Milana - AI-компаньон с памятью, расписанием, мультимодальностью и автономными действиями.</b>
</p>

## Milana в действии

### Память и естественный диалог

<p align="center">
  <img src="docs/assets/memory.webp" alt="Milana memory and natural dialogue" width="100%">
</p>

Milana хранит контекст отдельных диалогов, помнит важные факты, учитывает расписание и использует накопленную память в следующих разговорах.

### Фото, голосовые, файлы и стикеры

<p align="center">
  <img src="docs/assets/multimodal.webp" alt="Milana multimodal Telegram interaction" width="100%">
</p>

Бот умеет работать с фотографиями, голосовыми сообщениями, поддерживаемыми видео, файлами и стикерами, сохраняя естественный стиль общения.

### Панель управления и состояние сервиса

<p align="center">
  <img src="docs/assets/dashboard.webp" alt="Milana control panel" width="100%">
</p>

Локальная панель показывает состояние сервиса, Telegram-хоста, очереди, расписания, памяти и последних событий, а также предоставляет основные действия управления.

{END}
"""


def main() -> None:
    text = README.read_text(encoding="utf-8")
    if START in text:
        return

    first_line_end = text.find("\n")
    if first_line_end == -1:
        updated = text + "\n\n" + SHOWCASE
    else:
        updated = text[: first_line_end + 1] + "\n" + SHOWCASE + "\n" + text[first_line_end + 1 :]

    README.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
