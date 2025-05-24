import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from aiogram.types import Message as AiogramMessage
from pyrogram.types import Audio, Document
from pyrogram.types import Message as PyrogramMessage
from pyrogram.types import Photo, Video, VideoNote, Voice

PyrogramMedia = Union[Document, Audio, Video, Voice, VideoNote, Photo]


@lru_cache
async def get_pyrogram_client(
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    bot_token: Optional[str] = None,
):
    # telegram bot token,
    # api_id
    # api_hash
    if api_id is None:
        from botspot import get_dependency_manager

        deps = get_dependency_manager()
        api_id = deps.botspot_settings.telethon_manager.api_id
    if api_hash is None:
        from botspot import get_dependency_manager

        deps = get_dependency_manager()
        assert deps.botspot_settings.telethon_manager.api_hash is not None
        api_hash = deps.botspot_settings.telethon_manager.api_hash.get_secret_value()
    if bot_token is None:
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    from pyrogram import Client

    assert api_id is not None
    assert api_hash is not None
    assert bot_token is not None

    pyrogram_client = Client(
        "telegram_downloader", api_id=api_id, api_hash=api_hash, bot_token=bot_token
    )
    await pyrogram_client.start()
    return pyrogram_client


def get_media_and_media_type(msg: PyrogramMessage) -> tuple[PyrogramMedia, str, str]:
    if msg.document:
        ext = msg.document.file_name.split(".")[-1]
        return msg.document, "document", ext
    if msg.audio:
        ext = msg.audio.file_name.split(".")[-1]
        return msg.audio, "audio", ext
    if msg.video:
        ext = msg.video.file_name.split(".")[-1]
        return msg.video, "video", ext
    if msg.voice:
        ext = "ogg"
        return msg.voice, "voice", ext
    if msg.video_note:
        ext = "mp4"
        return msg.video_note, "video_note", ext
    if msg.photo:
        ext = "jpg"
        return msg.photo, "photo", ext
    raise ValueError(f"Unknown media type: {msg}")


def extract_file_name_from_pyrogram_message(
    msg: PyrogramMessage, use_original_file_name: bool = True
) -> str:
    media, media_type, ext = get_media_and_media_type(msg)
    if (not use_original_file_name) or (media_type in ("voice", "video_note", "photo")):
        short_id = media.file_id[-12:]
        return f"{media_type}_{short_id}.{ext}"
    assert isinstance(media, (Document, Audio, Video))
    return media.file_name


async def download_file_1(
    message: AiogramMessage,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
):
    """
    Returns: path to a file on disk
    """
    message_id = message.message_id
    assert message.from_user is not None
    username = message.from_user.username

    return await download_file_2(
        message_id, username, target_dir, file_name, use_original_file_name
    )


async def download_file_2(
    message_id,
    username,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    bot_token: Optional[str] = None,
):
    pyrogram_client = await get_pyrogram_client(
        api_id=api_id, api_hash=api_hash, bot_token=bot_token
    )

    pyrogram_message = await pyrogram_client.get_messages(
        username, message_ids=message_id
    )
    assert not isinstance(pyrogram_message, list)

    if target_dir is None:
        target_dir = Path(".")

    # todo: figure out the file name - get from attachment or smth?
    if file_name is None:
        file_name = extract_file_name_from_pyrogram_message(
            pyrogram_message, use_original_file_name
        )

    file_path = target_dir / file_name
    await pyrogram_message.download(file_name=str(file_path))

    return file_path


if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    load_dotenv()

    p = Path(__file__).parent / "mocking_pyrogram_message_types" / "sample_messages"
    message = pickle.load(open(p / "sample_photo_attached.pkl", "rb"))

    api_id = os.getenv("TELEGRAM_API_ID")
    if api_id is not None:
        api_id = int(api_id)
    
    res = asyncio.run(
        download_file_2(
            message.id,
            message.from_user.username,
            api_id=api_id,
            api_hash=os.getenv("TELEGRAM_API_HASH"),
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        )
    )
    print(res)
