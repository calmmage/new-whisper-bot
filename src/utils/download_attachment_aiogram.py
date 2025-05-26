from pathlib import Path
from typing import BinaryIO, Optional, Union

from aiogram.types import Message as AiogramMessage
from aiogram.types import Audio, Video, Document

# Attachment = Union[Audio, Video, Document, Voice, VideoNote, PhotoSize]
from botspot.utils.unsorted import Attachment


def get_media_and_media_type(msg: AiogramMessage) -> tuple[Attachment, str, str]:
    if msg.document:
        if msg.document.file_name is None:
            assert msg.document.mime_type is not None
            ext = msg.document.mime_type.split("/")[-1]
        else:
            ext = msg.document.file_name.split(".")[-1]
        return msg.document, "document", ext
    if msg.audio:
        if msg.audio.file_name is None:
            assert msg.audio.mime_type is not None
            ext = msg.audio.mime_type.split("/")[-1]
        else:
            ext = msg.audio.file_name.split(".")[-1]
        return msg.audio, "audio", ext
    if msg.video:
        if msg.video.file_name is None:
            assert msg.video.mime_type is not None
            ext = msg.video.mime_type.split("/")[-1]
        else:
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
        return msg.photo[0], "photo", ext
    raise ValueError(f"Unknown media type: {msg}")


def extract_file_name_from_aiogram_message(
    msg: AiogramMessage, use_original_file_name: bool = True
) -> str:
    media, media_type, ext = get_media_and_media_type(msg)
    use_name = use_original_file_name
    if media_type in ("voice", "video_note", "photo"):
        use_name = False
    else:
        assert isinstance(media, (Audio, Video, Document))
        if media.file_name is None:
            use_name = False
    if not use_name:
        short_id = media.file_id[-12:]
        return f"{media_type}_{short_id}.{ext}"
    assert isinstance(media, (Document, Audio, Video))
    assert media.file_name is not None

    return media.file_name


async def download_file_aiogram(
    message: AiogramMessage,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
    in_memory: bool = False,
):
    if in_memory:
        destination = None
    else:
        # todo: add support for use_original_file_name
        if target_dir is None:
            # what is the default downloads dir? just generate whatever?
            target_dir = Path("downloads")
        if file_name is None:
            # todo: grab the logic from pyrogram?
            file_name = extract_file_name_from_aiogram_message(
                message, use_original_file_name
            )
        destination = target_dir / file_name

    return await _download_file_aiogram(
        message=message,
        destination=destination,
    )


async def _download_file_aiogram(
    message: AiogramMessage,
    destination: Optional[Union[BinaryIO, Path, str]] = None,
) -> Union[BinaryIO, Path]:
    """

    If you want to automatically create destination (:class:`io.BytesIO`) use default
    value of destination and handle result of this method.

    :param destination:
        Filename, file path or instance of :class:`io.IOBase`. For e.g. :class:`io.BytesIO`, defaults to None
    """

    from botspot.utils.unsorted import get_message_attachments

    attachments = get_message_attachments(message)
    assert len(attachments) == 1

    attachment = attachments[0]

    from botspot.utils.deps_getters import get_bot

    bot = get_bot()

    file_id = attachment.file_id
    file = await bot.get_file(file_id)
    assert file and file.file_path, "File not found"

    result = await bot.download_file(file.file_path, destination=destination)

    if result is None:
        assert isinstance(destination, (Path, str))

        return Path(destination)
    else:
        return result
