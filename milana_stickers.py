"""On-demand visual sticker picker and durable Telegram sticker references."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps
from telethon.tl import functions, types


ANIMATED_STICKER_MIME_TYPE = "application/x-tgsticker"
VIDEO_STICKER_MIME_TYPE = "video/webm"
MAX_STICKER_TOOL_ROUNDS = 12
CONTACT_SHEET_COLUMNS = 5
CONTACT_SHEET_ROWS = 4
CONTACT_SHEET_TILE_WIDTH = 180
CONTACT_SHEET_TILE_HEIGHT = 190


OPEN_STICKER_PICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "open_sticker_picker",
    "description": (
        "Открыть внутренний визуальный выбор стикеров Миланы. Сначала вызови с "
        "pack_id=null, чтобы увидеть все установленные наборы. Затем вызови ещё раз "
        "с выбранным pack_id, чтобы увидеть каждый стикер этого набора. Каталог не "
        "показывается собеседнику."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "pack_id": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Идентификатор P001 из индекса либо null для индекса.",
            }
        },
        "required": ["pack_id"],
        "additionalProperties": False,
    },
}

SEND_STICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "send_sticker",
    "description": (
        "Выбрать для немедленной отправки стикер, уже показанный через "
        "open_sticker_picker в текущем ответе. Обычно выбирай один стикер."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"sticker_id": {"type": "string"}},
        "required": ["sticker_id"],
        "additionalProperties": False,
    },
}

SCHEDULE_STICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "schedule_sticker",
    "description": (
        "Поставить уже показанный стикер в отложенную отправку текущему собеседнику. "
        "Используй только для будущей отправки; delay_seconds считается от текущего момента."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "sticker_id": {"type": "string"},
            "delay_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 31_536_000,
            },
        },
        "required": ["sticker_id", "delay_seconds"],
        "additionalProperties": False,
    },
}

STICKER_TOOLS = (
    OPEN_STICKER_PICKER_TOOL,
    SEND_STICKER_TOOL,
    SCHEDULE_STICKER_TOOL,
)

STICKER_SKILL_INSTRUCTIONS = (
    "У тебя есть отдельный внутренний навык стикеров. Если ты сама хочешь отправить "
    "стикер, сначала вызови open_sticker_picker с pack_id=null: только после этого "
    "ты увидишь установленные наборы. Открой выбранный набор той же командой и "
    "осмысленно выбери стикер по контактным листам. Никогда не выдумывай pack_id или "
    "sticker_id и не выбирай непоказанный стикер. Для немедленной отправки вызови "
    "send_sticker, для отправки позже — schedule_sticker. Каталог служебный и не "
    "отображается собеседнику. Ты можешь сочетать текст и стикеры, но обычно одного "
    "стикера достаточно; серию используй только когда она действительно уместна."
)


@dataclass(frozen=True)
class StickerReference:
    set_id: int
    set_access_hash: int
    set_short_name: str
    document_id: int
    pack_title: str
    emoji: str

    @property
    def description(self) -> str:
        emoji = f"; эмодзи: {self.emoji}" if self.emoji else ""
        return f"[стикер из набора «{self.pack_title}»{emoji}]"


@dataclass(frozen=True)
class StickerChoice:
    reference: StickerReference
    document: Any


@dataclass(frozen=True)
class StagedScheduledSticker:
    delay_seconds: int
    choice: StickerChoice


@dataclass(frozen=True)
class StickerPickerOutput:
    content: tuple[dict[str, Any], ...]

    @classmethod
    def text(cls, payload: dict[str, Any]) -> "StickerPickerOutput":
        return cls(
            (
                {
                    "type": "input_text",
                    "text": json.dumps(payload, ensure_ascii=False),
                },
            )
        )


def _document_emoji(document: Any, packs: tuple[Any, ...]) -> str:
    for attribute in tuple(getattr(document, "attributes", None) or ()):
        if isinstance(attribute, types.DocumentAttributeSticker):
            return str(getattr(attribute, "alt", "") or "").strip()
    document_id = getattr(document, "id", None)
    for pack in packs:
        if document_id in tuple(getattr(pack, "documents", None) or ()):
            return str(getattr(pack, "emoticon", "") or "").strip()
    return ""


def _placeholder(label: str) -> Image.Image:
    image = Image.new("RGBA", (144, 144), (238, 238, 238, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 141, 141), outline=(150, 150, 150, 255), width=3)
    draw.line((18, 18, 126, 126), fill=(180, 180, 180, 255), width=4)
    draw.line((126, 18, 18, 126), fill=(180, 180, 180, 255), width=4)
    draw.text((8, 116), label[:18], fill=(50, 50, 50, 255))
    return image


def _open_preview(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as source:
        source.seek(0)
        return ImageOps.contain(source.convert("RGBA"), (144, 144))


def _contact_sheets(
    entries: list[tuple[str, Image.Image]],
) -> tuple[bytes, ...]:
    per_sheet = CONTACT_SHEET_COLUMNS * CONTACT_SHEET_ROWS
    result: list[bytes] = []
    for offset in range(0, len(entries), per_sheet):
        page = entries[offset : offset + per_sheet]
        sheet = Image.new(
            "RGBA",
            (
                CONTACT_SHEET_COLUMNS * CONTACT_SHEET_TILE_WIDTH,
                CONTACT_SHEET_ROWS * CONTACT_SHEET_TILE_HEIGHT,
            ),
            (250, 250, 250, 255),
        )
        draw = ImageDraw.Draw(sheet)
        for index, (label, preview) in enumerate(page):
            column = index % CONTACT_SHEET_COLUMNS
            row = index // CONTACT_SHEET_COLUMNS
            left = column * CONTACT_SHEET_TILE_WIDTH
            top = row * CONTACT_SHEET_TILE_HEIGHT
            draw.rectangle(
                (left + 3, top + 3, left + 176, top + 186),
                fill=(255, 255, 255, 255),
                outline=(210, 210, 210, 255),
                width=2,
            )
            fitted = ImageOps.contain(preview.convert("RGBA"), (150, 150))
            x = left + (CONTACT_SHEET_TILE_WIDTH - fitted.width) // 2
            y = top + 8 + (150 - fitted.height) // 2
            sheet.alpha_composite(fitted, (x, y))
            draw.text((left + 9, top + 166), label, fill=(20, 20, 20, 255))
        output = io.BytesIO()
        sheet.convert("RGB").save(output, format="PNG", optimize=True)
        result.append(output.getvalue())
    return tuple(result)


class MilanaStickerSkill:
    """Shared cached catalog used by reply, initiative and pulse flows."""

    def __init__(
        self,
        client: Any,
        *,
        animated_renderer: Callable[[bytes, str | None], bytes],
    ) -> None:
        self.client = client
        self.animated_renderer = animated_renderer
        self._all_hash = 0
        self._sets: tuple[Any, ...] = ()
        self._set_cache: dict[int, tuple[int, Any]] = {}
        self._preview_cache: dict[int, Image.Image] = {}

    def new_session(self) -> "StickerPickerSession":
        return StickerPickerSession(self)

    async def _refresh_sets(self) -> tuple[Any, ...]:
        result = await self.client(
            functions.messages.GetAllStickersRequest(hash=self._all_hash)
        )
        if isinstance(result, types.messages.AllStickersNotModified):
            return self._sets
        sets = tuple(
            item
            for item in tuple(getattr(result, "sets", None) or ())
            if not bool(getattr(item, "masks", False))
            and not bool(getattr(item, "emojis", False))
        )
        self._all_hash = int(getattr(result, "hash", 0) or 0)
        self._sets = sets
        valid_ids = {int(item.id) for item in sets}
        self._set_cache = {
            key: value for key, value in self._set_cache.items() if key in valid_ids
        }
        return sets

    async def _load_set(self, sticker_set: Any) -> Any:
        set_id = int(sticker_set.id)
        set_hash = int(getattr(sticker_set, "hash", 0) or 0)
        cached = self._set_cache.get(set_id)
        if cached is not None and cached[0] == set_hash:
            return cached[1]
        result = await self.client(
            functions.messages.GetStickerSetRequest(
                stickerset=types.InputStickerSetID(
                    id=set_id,
                    access_hash=int(sticker_set.access_hash),
                ),
                hash=0,
            )
        )
        if isinstance(result, types.messages.StickerSetNotModified):
            if cached is None:
                raise ValueError("Telegram не вернул содержимое стикерпака")
            return cached[1]
        self._set_cache[set_id] = (set_hash, result)
        return result

    async def _preview(self, document: Any, label: str) -> Image.Image:
        document_id = int(getattr(document, "id", 0) or 0)
        cached = self._preview_cache.get(document_id)
        if cached is not None:
            return cached.copy()
        thumbs = tuple(getattr(document, "thumbs", None) or ())
        raster_thumbs = [
            thumb for thumb in thumbs if not isinstance(thumb, types.PhotoPathSize)
        ]
        if raster_thumbs:
            try:
                data = await self.client.download_media(
                    document, file=bytes, thumb=raster_thumbs[-1]
                )
                if isinstance(data, bytes) and data:
                    preview = await asyncio.to_thread(_open_preview, data)
                    self._preview_cache[document_id] = preview.copy()
                    return preview
            except Exception:  # noqa: BLE001 - a placeholder keeps the full catalog usable
                pass
        try:
            data = await self.client.download_media(document, file=bytes)
            if not isinstance(data, bytes) or not data:
                raise ValueError("пустой документ")
            mime_type = str(getattr(document, "mime_type", "") or "")
            if mime_type in {ANIMATED_STICKER_MIME_TYPE, VIDEO_STICKER_MIME_TYPE}:
                data = await asyncio.to_thread(
                    self.animated_renderer, data, mime_type
                )
            preview = await asyncio.to_thread(_open_preview, data)
            self._preview_cache[document_id] = preview.copy()
            return preview
        except Exception:  # noqa: BLE001 - preserve every sticker as a selectable tile
            preview = _placeholder(label)
            self._preview_cache[document_id] = preview.copy()
            return preview

    async def resolve_reference(self, reference: StickerReference) -> StickerChoice:
        result = await self.client(
            functions.messages.GetStickerSetRequest(
                stickerset=types.InputStickerSetID(
                    id=reference.set_id,
                    access_hash=reference.set_access_hash,
                ),
                hash=0,
            )
        )
        documents = tuple(getattr(result, "documents", None) or ())
        document = next(
            (
                item
                for item in documents
                if int(getattr(item, "id", -1)) == reference.document_id
            ),
            None,
        )
        if document is None:
            raise ValueError("Отложенный стикер больше не найден в наборе")
        return StickerChoice(reference, document)


class StickerPickerSession:
    """Ephemeral IDs and viewed-sticker authorization for one model generation."""

    def __init__(self, skill: MilanaStickerSkill) -> None:
        self.skill = skill
        self._packs: dict[str, Any] = {}
        self._viewed: dict[str, StickerChoice] = {}

    @staticmethod
    def _image_content(png: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(png).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{encoded}",
            "detail": "original",
        }

    async def open(self, pack_id: Any = None) -> StickerPickerOutput:
        if pack_id is None:
            sets = await self.skill._refresh_sets()
            self._packs = {
                f"P{index:03d}": item for index, item in enumerate(sets, start=1)
            }
            if not self._packs:
                return StickerPickerOutput.text(
                    {"status": "empty", "view": "packs", "packs": []}
                )

            semaphore = asyncio.Semaphore(4)

            async def cover(item: tuple[str, Any]) -> tuple[str, Image.Image]:
                local_id, sticker_set = item
                async with semaphore:
                    try:
                        full_set = await self.skill._load_set(sticker_set)
                        documents = tuple(getattr(full_set, "documents", None) or ())
                        if not documents:
                            raise ValueError("пустой набор")
                        preview = await self.skill._preview(documents[0], local_id)
                    except Exception:  # noqa: BLE001 - one bad set must not hide others
                        preview = _placeholder(local_id)
                    return local_id, preview

            entries = list(await asyncio.gather(*(cover(item) for item in self._packs.items())))
            sheets = await asyncio.to_thread(_contact_sheets, entries)
            payload = {
                "status": "ok",
                "view": "packs",
                "packs": [
                    {
                        "pack_id": local_id,
                        "title": str(getattr(sticker_set, "title", "") or ""),
                        "count": int(getattr(sticker_set, "count", 0) or 0),
                    }
                    for local_id, sticker_set in self._packs.items()
                ],
            }
            return StickerPickerOutput(
                (
                    {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)},
                    *(self._image_content(sheet) for sheet in sheets),
                )
            )

        if not isinstance(pack_id, str) or pack_id not in self._packs:
            return StickerPickerOutput.text(
                {
                    "status": "error",
                    "message": "Сначала открой индекс и выбери существующий pack_id.",
                }
            )
        sticker_set = self._packs[pack_id]
        full_set = await self.skill._load_set(sticker_set)
        documents = tuple(getattr(full_set, "documents", None) or ())
        packs = tuple(getattr(full_set, "packs", None) or ())
        items: list[dict[str, str]] = []
        preview_documents: list[tuple[str, Any]] = []
        for index, document in enumerate(documents, start=1):
            sticker_id = f"{pack_id}:S{index:03d}"
            emoji = _document_emoji(document, packs)
            reference = StickerReference(
                set_id=int(sticker_set.id),
                set_access_hash=int(sticker_set.access_hash),
                set_short_name=str(getattr(sticker_set, "short_name", "") or ""),
                document_id=int(document.id),
                pack_title=str(getattr(sticker_set, "title", "") or pack_id),
                emoji=emoji,
            )
            self._viewed[sticker_id] = StickerChoice(reference, document)
            preview_documents.append((sticker_id.split(":")[-1], document))
            items.append({"sticker_id": sticker_id, "emoji": emoji})
        semaphore = asyncio.Semaphore(4)

        async def sticker_preview(item: tuple[str, Any]) -> tuple[str, Image.Image]:
            label, document = item
            async with semaphore:
                return label, await self.skill._preview(document, label)

        entries = list(
            await asyncio.gather(*(sticker_preview(item) for item in preview_documents))
        )
        sheets = await asyncio.to_thread(_contact_sheets, entries)
        payload = {
            "status": "ok",
            "view": "pack",
            "pack_id": pack_id,
            "title": str(getattr(sticker_set, "title", "") or ""),
            "stickers": items,
        }
        return StickerPickerOutput(
            (
                {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)},
                *(self._image_content(sheet) for sheet in sheets),
            )
        )

    def choose(self, sticker_id: Any) -> StickerChoice:
        if not isinstance(sticker_id, str) or sticker_id not in self._viewed:
            raise ValueError(
                "Стикер не был показан в открытом наборе текущего выбора"
            )
        return self._viewed[sticker_id]
