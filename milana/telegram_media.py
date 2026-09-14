"""Telegram media detection, download, conversion and sticker rendering helpers."""

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


SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
SUPPORTED_GEMINI_VIDEO_MIME_TYPES = {
    "video/3gpp",
    "video/avi",
    "video/mov",
    "video/mp4",
    "video/mpeg",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/x-flv",
}
SUPPORTED_GEMINI_AUDIO_MIME_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/webm",
}
GEMINI_AUDIO_MIME_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/x-m4a": "audio/mp4",
    "audio/x-wav": "audio/wav",
}
GEMINI_VIDEO_MIME_ALIASES = {
    "video/quicktime": "video/mov",
    "video/x-msvideo": "video/avi",
    "video/x-ms-wmv": "video/wmv",
}
MAX_GEMINI_INLINE_VIDEO_BYTES = 20 * 1024 * 1024
MAX_GEMINI_INLINE_AUDIO_BYTES = 20 * 1024 * 1024
ANIMATED_STICKER_MIME_TYPE = "application/x-tgsticker"
VIDEO_STICKER_MIME_TYPE = "video/webm"


@dataclass(frozen=True)
class TelegramStickerInfo:
    description: str
    mime_type: str | None
    thumbnail: Any | None


def telegram_image_mime_type(event: Any) -> str | None:
    """Возвращает MIME-тип поддерживаемого изображения из Telegram-события."""
    message = getattr(event, "message", None)
    photo = getattr(event, "photo", None) or getattr(message, "photo", None)
    file_info = getattr(event, "file", None) or getattr(message, "file", None)
    mime_type = getattr(file_info, "mime_type", None)

    # Обычные Telegram-фото всегда отдаются как JPEG, даже если у File нет MIME.
    if photo is not None:
        return mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else "image/jpeg"
    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return mime_type
    return None


def telegram_video_mime_type(event: Any) -> str | None:
    """Возвращает поддерживаемый Gemini MIME обычного Telegram-видео."""
    message = getattr(event, "message", None)
    sticker = getattr(event, "sticker", None)
    if sticker is None and message is not None and message is not event:
        sticker = getattr(message, "sticker", None)
    if sticker is not None:
        return None

    file_info = getattr(event, "file", None)
    if file_info is None and message is not None and message is not event:
        file_info = getattr(message, "file", None)
    mime_type = getattr(file_info, "mime_type", None)
    if not isinstance(mime_type, str):
        return None
    normalized = GEMINI_VIDEO_MIME_ALIASES.get(mime_type.lower(), mime_type.lower())
    return normalized if normalized in SUPPORTED_GEMINI_VIDEO_MIME_TYPES else None


def telegram_voice_mime_type(event: Any) -> str | None:
    """Возвращает поддерживаемый Gemini MIME голосового сообщения Telegram."""
    message = getattr(event, "message", None)
    voice = getattr(event, "voice", None)
    if voice is None and message is not None and message is not event:
        voice = getattr(message, "voice", None)
    if not voice:
        return None

    file_info = getattr(event, "file", None)
    if file_info is None and message is not None and message is not event:
        file_info = getattr(message, "file", None)
    mime_type = getattr(file_info, "mime_type", None)
    if not isinstance(mime_type, str):
        # Telegram voice notes are OGG/Opus even when File has no MIME metadata.
        return "audio/ogg"
    normalized = GEMINI_AUDIO_MIME_ALIASES.get(mime_type.lower(), mime_type.lower())
    return normalized if normalized in SUPPORTED_GEMINI_AUDIO_MIME_TYPES else None


def telegram_sticker_info(event: Any) -> TelegramStickerInfo | None:
    """Возвращает описание стикера и доступное растровое превью."""
    message = getattr(event, "message", None)
    sticker = getattr(event, "sticker", None)
    if sticker is None and message is not None and message is not event:
        sticker = getattr(message, "sticker", None)

    file_info = getattr(event, "file", None)
    if file_info is None and message is not None and message is not event:
        file_info = getattr(message, "file", None)
    emoji = getattr(file_info, "emoji", None)
    if sticker is None and emoji is None:
        return None

    mime_type = getattr(file_info, "mime_type", None)
    if mime_type == ANIMATED_STICKER_MIME_TYPE:
        kind = "анимированный стикер"
    elif mime_type == VIDEO_STICKER_MIME_TYPE:
        kind = "видеостикер"
    else:
        kind = "стикер"

    normalized_emoji = emoji.strip() if isinstance(emoji, str) else ""
    description = (
        f"[{kind}; эмодзи: {normalized_emoji}]"
        if normalized_emoji
        else f"[{kind}]"
    )

    thumbnail: Any | None = None
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES and sticker is not None:
        thumbs = tuple(getattr(sticker, "thumbs", None) or ())
        unsupported_thumb_types = (
            types.PhotoPathSize,
            types.PhotoSizeEmpty,
            # Это крошечный размытый placeholder. Исходный TGS/WebM даст
            # модели заметно более информативный кадр.
            types.PhotoStrippedSize,
            types.VideoSize,
        )
        raster_thumbs = [
            candidate
            for candidate in thumbs
            if not isinstance(candidate, unsupported_thumb_types)
        ]

        def thumbnail_rank(candidate: Any) -> tuple[int, int]:
            if isinstance(candidate, types.PhotoCachedSize):
                return (1, len(getattr(candidate, "bytes", b"") or b""))
            if isinstance(candidate, types.PhotoSizeProgressive):
                return (1, max(getattr(candidate, "sizes", ()) or (0,)))
            size = getattr(candidate, "size", 0)
            normalized_size = size if isinstance(size, int) and size >= 0 else 0
            return (1, normalized_size)

        if raster_thumbs:
            largest = max(raster_thumbs, key=thumbnail_rank)
            thumbnail = getattr(largest, "type", None) or largest

    return TelegramStickerInfo(
        description=description,
        mime_type=mime_type,
        thumbnail=thumbnail,
    )


def image_mime_type_from_bytes(image_bytes: bytes) -> str | None:
    """Определяет поддерживаемый MIME изображения по сигнатуре файла."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


async def telegram_image_data_url(
    event: Any,
    mime_type: str | None,
    *,
    thumbnail: Any | None = None,
) -> str:
    """Скачивает Telegram-изображение в память и кодирует для Responses API."""
    message = getattr(event, "message", None)
    download_media = getattr(message, "download_media", None)
    if not callable(download_media):
        download_media = getattr(event, "download_media", None)
    if not callable(download_media):
        raise ValueError("Telegram не предоставил способ скачать изображение")

    download_kwargs: dict[str, Any] = {"file": bytes}
    if thumbnail is not None:
        download_kwargs["thumb"] = thumbnail
    image_bytes = await download_media(**download_kwargs)
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Не удалось скачать изображение из Telegram")
    detected_mime_type = image_mime_type_from_bytes(image_bytes)
    resolved_mime_type = detected_mime_type or mime_type
    if resolved_mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError("Telegram вернул превью в неподдерживаемом формате")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{resolved_mime_type};base64,{encoded}"


async def telegram_video_data_url(event: Any, mime_type: str) -> str:
    """Скачивает небольшое Telegram-видео и кодирует его для Gemini-адаптера."""
    if mime_type not in SUPPORTED_GEMINI_VIDEO_MIME_TYPES:
        raise ValueError(f"Неподдерживаемый Gemini формат видео: {mime_type}")

    message = getattr(event, "message", None)
    file_info = getattr(event, "file", None)
    if file_info is None and message is not None and message is not event:
        file_info = getattr(message, "file", None)
    declared_size = getattr(file_info, "size", None)
    if (
        isinstance(declared_size, int)
        and declared_size >= MAX_GEMINI_INLINE_VIDEO_BYTES
    ):
        raise ValueError(
            "Видео слишком большое для прямой передачи Gemini "
            f"({declared_size} байт; лимит меньше {MAX_GEMINI_INLINE_VIDEO_BYTES})"
        )

    download_media = getattr(message, "download_media", None)
    if not callable(download_media):
        download_media = getattr(event, "download_media", None)
    if not callable(download_media):
        raise ValueError("Telegram не предоставил способ скачать видео")

    video_bytes = await download_media(file=bytes)
    if not isinstance(video_bytes, bytes) or not video_bytes:
        raise ValueError("Не удалось скачать видео из Telegram")
    if len(video_bytes) >= MAX_GEMINI_INLINE_VIDEO_BYTES:
        raise ValueError(
            "Видео слишком большое для прямой передачи Gemini "
            f"({len(video_bytes)} байт; лимит меньше {MAX_GEMINI_INLINE_VIDEO_BYTES})"
        )
    encoded = base64.b64encode(video_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def convert_gif_to_mp4(gif_bytes: bytes) -> bytes:
    """Преобразует GIF-анимацию в поддерживаемое Gemini MP4-видео."""
    if not gif_bytes.startswith((b"GIF87a", b"GIF89a")):
        raise ValueError("Telegram вернул данные, которые не являются GIF")

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_executable = get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError) as exc:
        raise ValueError(f"FFmpeg для преобразования GIF недоступен: {exc}") from exc

    with TemporaryDirectory(prefix="milana-gif-") as directory:
        gif_path = Path(directory) / "animation.gif"
        video_path = Path(directory) / "animation.mp4"
        gif_path.write_bytes(gif_bytes)
        try:
            completed = subprocess.run(
                [
                    ffmpeg_executable,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(gif_path),
                    "-map_metadata",
                    "-1",
                    "-an",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                capture_output=True,
                timeout=30,
                check=False,
                **hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"FFmpeg не смог преобразовать GIF: {exc}") from exc
        if completed.returncode != 0:
            details = completed.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(
                "FFmpeg не смог преобразовать GIF"
                + (f": {details[-500:]}" if details else "")
            )
        try:
            video_bytes = video_path.read_bytes()
        except OSError as exc:
            raise ValueError("FFmpeg не создал MP4 из GIF") from exc

    if not video_bytes:
        raise ValueError("FFmpeg создал пустое MP4 из GIF")
    if len(video_bytes) >= MAX_GEMINI_INLINE_VIDEO_BYTES:
        raise ValueError(
            "GIF после преобразования слишком большая для прямой передачи Gemini "
            f"({len(video_bytes)} байт; лимит меньше {MAX_GEMINI_INLINE_VIDEO_BYTES})"
        )
    return video_bytes


async def telegram_voice_data_url(event: Any, mime_type: str) -> str:
    """Скачивает голосовое Telegram и кодирует его для Gemini-адаптера."""
    if mime_type not in SUPPORTED_GEMINI_AUDIO_MIME_TYPES:
        raise ValueError(f"Неподдерживаемый Gemini формат аудио: {mime_type}")

    message = getattr(event, "message", None)
    file_info = getattr(event, "file", None)
    if file_info is None and message is not None and message is not event:
        file_info = getattr(message, "file", None)
    declared_size = getattr(file_info, "size", None)
    if (
        isinstance(declared_size, int)
        and declared_size >= MAX_GEMINI_INLINE_AUDIO_BYTES
    ):
        raise ValueError(
            "Голосовое сообщение слишком большое для прямой передачи Gemini "
            f"({declared_size} байт; лимит меньше {MAX_GEMINI_INLINE_AUDIO_BYTES})"
        )

    download_media = getattr(message, "download_media", None)
    if not callable(download_media):
        download_media = getattr(event, "download_media", None)
    if not callable(download_media):
        raise ValueError("Telegram не предоставил способ скачать голосовое сообщение")

    audio_bytes = await download_media(file=bytes)
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        raise ValueError("Не удалось скачать голосовое сообщение из Telegram")
    if len(audio_bytes) >= MAX_GEMINI_INLINE_AUDIO_BYTES:
        raise ValueError(
            "Голосовое сообщение слишком большое для прямой передачи Gemini "
            f"({len(audio_bytes)} байт; лимит меньше {MAX_GEMINI_INLINE_AUDIO_BYTES})"
        )
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _pillow_image_png_bytes(image: Any) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    png_bytes = output.getvalue()
    if not png_bytes:
        raise ValueError("Рендерер вернул пустое изображение")
    return png_bytes


def _render_tgs_sticker_png(sticker_bytes: bytes) -> bytes:
    from rlottie_python import LottieAnimation

    animation_json = gzip.decompress(sticker_bytes).decode("utf-8")
    with LottieAnimation.from_data(data=animation_json) as animation:
        total_frames = int(animation.lottie_animation_get_totalframe())
        frame_number = max(0, total_frames // 2)
        image = animation.render_pillow_frame(frame_num=frame_number)
        return _pillow_image_png_bytes(image)


def _render_webm_sticker_png(sticker_bytes: bytes) -> bytes:
    from imageio_ffmpeg import get_ffmpeg_exe
    from PIL import Image

    with TemporaryDirectory(prefix="milana-sticker-") as directory:
        video_path = Path(directory) / "sticker.webm"
        video_path.write_bytes(sticker_bytes)
        try:
            completed = subprocess.run(
                [
                    get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    # Нативный декодер VP9 теряет alpha plane WebM-стикеров.
                    "-c:v",
                    "libvpx-vp9",
                    "-i",
                    str(video_path),
                    "-map_metadata",
                    "-1",
                    # Берём характерный кадр из первых 30, а не пустой стартовый.
                    "-vf",
                    "thumbnail=30",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                check=False,
                **hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(
                f"FFmpeg не смог отрендерить WebM-стикер: {exc}"
            ) from exc

    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            "FFmpeg не смог отрендерить WebM-стикер"
            + (f": {details[-500:]}" if details else "")
        )
    if not completed.stdout:
        raise ValueError("FFmpeg не вернул кадр WebM-стикера")

    try:
        with Image.open(io.BytesIO(completed.stdout)) as decoded:
            decoded.load()
            image = decoded.convert("RGBA")
    except (OSError, ValueError) as exc:
        raise ValueError("FFmpeg вернул повреждённый PNG-кадр") from exc
    return _pillow_image_png_bytes(image)


def render_sticker_png(sticker_bytes: bytes, mime_type: str | None) -> bytes:
    """Рендерит репрезентативный кадр TGS/WebM-стикера в PNG."""
    renderer: Callable[[bytes], bytes]
    if mime_type == ANIMATED_STICKER_MIME_TYPE:
        renderer = _render_tgs_sticker_png
    elif mime_type == VIDEO_STICKER_MIME_TYPE:
        renderer = _render_webm_sticker_png
    else:
        raise ValueError(f"Неподдерживаемый формат стикера: {mime_type}")

    try:
        return renderer(sticker_bytes)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Не удалось отрендерить стикер {mime_type}: {exc}") from exc

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
