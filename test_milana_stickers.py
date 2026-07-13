from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image
from telethon.tl import functions, types

from milana_stickers import MilanaStickerSkill, StickerReference


def webp_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(output, format="WEBP")
    return output.getvalue()


def document(document_id: int, emoji: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=document_id,
        mime_type="image/webp",
        thumbs=(),
        attributes=(
            types.DocumentAttributeSticker(
                alt=emoji,
                stickerset=types.InputStickerSetEmpty(),
            ),
        ),
    )


class FakeTelegramClient:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.downloads: list[int] = []
        self.regular = SimpleNamespace(
            id=10,
            access_hash=20,
            title="Обычный набор",
            short_name="regular",
            count=23,
            hash=30,
            masks=False,
            emojis=False,
        )
        self.mask = SimpleNamespace(
            id=11,
            access_hash=21,
            title="Маски",
            short_name="masks",
            count=1,
            hash=31,
            masks=True,
            emojis=False,
        )
        self.custom_emoji = SimpleNamespace(
            id=12,
            access_hash=22,
            title="Эмодзи",
            short_name="emoji",
            count=1,
            hash=32,
            masks=False,
            emojis=True,
        )
        self.documents = tuple(document(1000 + index, "🙂") for index in range(23))
        self.not_modified = False

    async def __call__(self, request: object) -> object:
        self.requests.append(request)
        if isinstance(request, functions.messages.GetAllStickersRequest):
            if self.not_modified and request.hash == 777:
                return types.messages.AllStickersNotModified()
            return types.messages.AllStickers(
                hash=777,
                sets=[self.regular, self.mask, self.custom_emoji],
            )
        if isinstance(request, functions.messages.GetStickerSetRequest):
            return SimpleNamespace(
                set=self.regular,
                documents=self.documents,
                packs=(),
            )
        raise AssertionError(type(request))

    async def download_media(self, value, *, file, thumb=None):
        self.downloads.append(int(value.id))
        return webp_bytes((40, int(value.id) % 255, 180))


class StickerSkillTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.client = FakeTelegramClient()
        self.skill = MilanaStickerSkill(
            self.client,
            animated_renderer=lambda data, mime: data,
        )

    async def test_picker_lists_only_regular_sets_and_every_sticker(self) -> None:
        session = self.skill.new_session()

        index = await session.open(None)
        index_payload = json.loads(index.content[0]["text"])
        self.assertEqual(
            index_payload["packs"],
            [{"pack_id": "P001", "title": "Обычный набор", "count": 23}],
        )
        self.assertEqual(len(index.content), 2)

        pack = await session.open("P001")
        pack_payload = json.loads(pack.content[0]["text"])
        self.assertEqual(len(pack_payload["stickers"]), 23)
        self.assertEqual(pack_payload["stickers"][0]["sticker_id"], "P001:S001")
        self.assertEqual(pack_payload["stickers"][-1]["sticker_id"], "P001:S023")
        self.assertEqual(len(pack.content), 3)  # text + two 20-tile contact sheets
        self.assertEqual(session.choose("P001:S023").document.id, 1022)

    async def test_unviewed_and_stale_ids_are_rejected(self) -> None:
        first = self.skill.new_session()
        await first.open(None)
        with self.assertRaises(ValueError):
            first.choose("P001:S001")

        await first.open("P001")
        self.assertEqual(first.choose("P001:S001").document.id, 1000)

        second = self.skill.new_session()
        with self.assertRaises(ValueError):
            second.choose("P001:S001")

    async def test_catalog_hash_is_reused_between_sessions(self) -> None:
        await self.skill.new_session().open(None)
        self.client.not_modified = True
        await self.skill.new_session().open(None)

        all_requests = [
            item
            for item in self.client.requests
            if isinstance(item, functions.messages.GetAllStickersRequest)
        ]
        self.assertEqual([item.hash for item in all_requests], [0, 777])

    async def test_resolve_reference_refreshes_document(self) -> None:
        reference = StickerReference(10, 20, "regular", 1005, "Обычный набор", "🙂")

        choice = await self.skill.resolve_reference(reference)

        self.assertEqual(choice.document.id, 1005)
        self.assertIn("Обычный набор", choice.reference.description)

    async def test_broken_preview_keeps_sticker_in_catalog(self) -> None:
        original_download = self.client.download_media

        async def download(value, *, file, thumb=None):
            if value.id == 1005:
                raise OSError("broken sticker")
            return await original_download(value, file=file, thumb=thumb)

        self.client.download_media = download
        session = self.skill.new_session()
        await session.open(None)

        pack = await session.open("P001")

        payload = json.loads(pack.content[0]["text"])
        self.assertEqual(len(payload["stickers"]), 23)
        self.assertEqual(session.choose("P001:S006").document.id, 1005)

    async def test_animated_document_uses_shared_renderer_and_cached_preview(self) -> None:
        self.client.documents[0].mime_type = "application/x-tgsticker"
        original_download = self.client.download_media

        async def download(value, *, file, thumb=None):
            if value.id == 1000:
                self.client.downloads.append(1000)
                return b"not-an-image"
            return await original_download(value, file=file, thumb=thumb)

        self.client.download_media = download
        renderer = MagicMock(return_value=webp_bytes((255, 0, 0)))
        skill = MilanaStickerSkill(self.client, animated_renderer=renderer)
        session = skill.new_session()

        await session.open(None)
        await session.open("P001")

        renderer.assert_called_once_with(b"not-an-image", "application/x-tgsticker")
        self.assertEqual(self.client.downloads.count(1000), 1)
