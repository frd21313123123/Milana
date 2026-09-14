"""One-shot safe refactor for extracting Telegram media helpers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "telegram_client.py"
MODULE_PATH = ROOT / "milana" / "telegram_media.py"

IMPORT_ANCHOR = "from milana_memory import (\n"
MEDIA_IMPORT = '''from milana.telegram_media import (
    ANIMATED_STICKER_MIME_TYPE,
    GEMINI_AUDIO_MIME_ALIASES,
    GEMINI_VIDEO_MIME_ALIASES,
    MAX_GEMINI_INLINE_AUDIO_BYTES,
    MAX_GEMINI_INLINE_VIDEO_BYTES,
    SUPPORTED_GEMINI_AUDIO_MIME_TYPES,
    SUPPORTED_GEMINI_VIDEO_MIME_TYPES,
    SUPPORTED_IMAGE_MIME_TYPES,
    VIDEO_STICKER_MIME_TYPE,
    TelegramStickerInfo,
    convert_gif_to_mp4,
    image_mime_type_from_bytes,
    render_sticker_png,
    telegram_image_data_url,
    telegram_image_mime_type,
    telegram_sticker_info,
    telegram_video_data_url,
    telegram_video_mime_type,
    telegram_voice_data_url,
    telegram_voice_mime_type,
)
'''

MODULE_HEADER = '''"""Telegram media detection, download, conversion and sticker rendering helpers."""

from __future__ import annotations

import base64
import gzip
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from telethon import types

from milana.subprocesses import hidden_subprocess_kwargs


'''

MODULE_EXPORTS = '''

__all__ = [
    "ANIMATED_STICKER_MIME_TYPE",
    "GEMINI_AUDIO_MIME_ALIASES",
    "GEMINI_VIDEO_MIME_ALIASES",
    "MAX_GEMINI_INLINE_AUDIO_BYTES",
    "MAX_GEMINI_INLINE_VIDEO_BYTES",
    "SUPPORTED_GEMINI_AUDIO_MIME_TYPES",
    "SUPPORTED_GEMINI_VIDEO_MIME_TYPES",
    "SUPPORTED_IMAGE_MIME_TYPES",
    "VIDEO_STICKER_MIME_TYPE",
    "TelegramStickerInfo",
    "convert_gif_to_mp4",
    "image_mime_type_from_bytes",
    "render_sticker_png",
    "telegram_image_data_url",
    "telegram_image_mime_type",
    "telegram_sticker_info",
    "telegram_video_data_url",
    "telegram_video_mime_type",
    "telegram_voice_data_url",
    "telegram_voice_mime_type",
]
'''


def extract(text: str, start_marker: str, end_marker: str) -> tuple[str, str]:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end], text[:start] + text[end:]


def main() -> None:
    text = CLIENT_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if "from milana.telegram_media import (" in text:
        raise SystemExit("telegram_client.py media helpers are already refactored")
    if IMPORT_ANCHOR not in text:
        raise RuntimeError("Telegram media import anchor not found")

    constants, text = extract(
        text,
        "SUPPORTED_IMAGE_MIME_TYPES = {",
        "SAFE_REACTIONS =",
    )
    sticker_info, text = extract(
        text,
        "@dataclass(frozen=True)\nclass TelegramStickerInfo:",
        "def load_ai_config() -> AIConfig:",
    )
    first_helpers, text = extract(
        text,
        "def telegram_image_mime_type(event: Any) -> str | None:",
        "async def telegram_gif_video_data_url(event: Any) -> str:",
    )
    second_helpers, text = extract(
        text,
        "async def telegram_voice_data_url(event: Any, mime_type: str) -> str:",
        "async def telegram_rendered_sticker_data_url(",
    )

    text = text.replace(IMPORT_ANCHOR, MEDIA_IMPORT + IMPORT_ANCHOR, 1)

    module_source = (
        MODULE_HEADER
        + constants.strip()
        + "\n\n\n"
        + sticker_info.strip()
        + "\n\n\n"
        + first_helpers.strip()
        + "\n\n\n"
        + second_helpers.strip()
        + MODULE_EXPORTS
    )

    required_client_names = (
        "convert_gif_to_mp4",
        "render_sticker_png",
        "telegram_image_mime_type",
        "telegram_video_mime_type",
        "telegram_voice_mime_type",
        "telegram_sticker_info",
        "telegram_image_data_url",
        "telegram_video_data_url",
        "telegram_voice_data_url",
    )
    for name in required_client_names:
        if name not in text:
            raise RuntimeError(f"compatibility export disappeared: {name}")

    # These wrappers intentionally stay in telegram_client.py so existing tests and
    # importers that monkey-patch the legacy names keep controlling the dependency.
    if "async def telegram_gif_video_data_url(event: Any) -> str:" not in text:
        raise RuntimeError("GIF compatibility wrapper disappeared")
    if "async def telegram_rendered_sticker_data_url(" not in text:
        raise RuntimeError("sticker render compatibility wrapper disappeared")

    new_lines = len(text.splitlines())
    removed = original_lines - new_lines
    if removed < 300:
        raise RuntimeError(f"media refactor removed only {removed} lines; expected >= 300")

    MODULE_PATH.write_text(module_source, encoding="utf-8")
    CLIENT_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted Telegram media helpers: telegram_client.py {original_lines} -> "
        f"{new_lines} lines; created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
