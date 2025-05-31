import asyncio
import os
import re
from pathlib import Path
from typing import Optional, Union, BinaryIO

from aiogram.types import Message as AiogramMessage
from loguru import logger
from pyrogram.types import Audio, Document
from pyrogram.types import Message as PyrogramMessage
from pyrogram.types import Photo, Video, VideoNote, Voice

PyrogramMedia = Union[Document, Audio, Video, Voice, VideoNote, Photo]

# Global client cache
_pyrogram_client = None


async def get_pyrogram_client(
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    bot_token: Optional[str] = None,
):
    global _pyrogram_client

    # Return existing client if available
    if _pyrogram_client is not None:
        return _pyrogram_client

    # telegram bot token, api_id, api_hash
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

    from pyrogram.client import Client

    assert api_id is not None
    assert api_hash is not None
    assert bot_token is not None

    _pyrogram_client = Client(
        "telegram_downloader", api_id=api_id, api_hash=api_hash, bot_token=bot_token
    )
    await _pyrogram_client.start()
    return _pyrogram_client


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


async def download_file_from_aiogram_message(
    message: AiogramMessage,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
    use_subprocess: bool = False,
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    bot_token: Optional[str] = None,
    in_memory: bool = False,
):
    """
    Download file from aiogram message.

    Args:
        message: Aiogram message object
        target_dir: Directory to save the file
        file_name: Custom file name (optional)
        use_original_file_name: Whether to use original file name
        use_subprocess: If True, use subprocess to avoid aiogram/pyrogram conflicts

    Returns: path to a file on disk
    """
    message_id = message.message_id
    assert message.from_user is not None
    username = message.from_user.username

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

    assert api_id is not None
    assert api_hash is not None
    assert bot_token is not None

    if use_subprocess:
        return await download_file_via_subprocess(
            message_id,
            username,
            api_id=api_id,
            api_hash=api_hash,
            bot_token=bot_token,
            target_dir=target_dir,
            file_name=file_name,
            use_original_file_name=use_original_file_name,
            in_memory=in_memory,
        )
    else:
        return await download_file_with_pyrogram(
            message_id, username, target_dir, file_name, use_original_file_name
        )


def _check_aiogram_running():
    """Check if aiogram is currently running and could conflict with pyrogram."""
    # Check if aiogram modules are imported and active

    # Check for common aiogram patterns in the current event loop
    try:
        import asyncio

        loop = asyncio.get_running_loop()

        # Check if there are tasks with aiogram-related names
        for task in asyncio.all_tasks(loop):
            task_name = str(task)
            if any(name in task_name.lower() for name in ["aiogram"]):
                logger.info(f"Found aiogram task: {task_name}")
                return True

    except RuntimeError:
        # No event loop running
        pass

    return False


async def download_file_with_pyrogram(
    message_id,
    username,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    bot_token: Optional[str] = None,
    check_aiogram: bool = True,
    in_memory: bool = False,
) -> Union[BinaryIO, Path]:
    # Check for aiogram conflicts
    if check_aiogram and _check_aiogram_running():
        raise RuntimeError(
            "Pyrogram is incompatible with aiogram when running in the same event loop. "
            "Use download_file_via_subprocess() instead to avoid conflicts."
        )

    pyrogram_client = await get_pyrogram_client(
        api_id=api_id, api_hash=api_hash, bot_token=bot_token
    )

    pyrogram_message = await pyrogram_client.get_messages(
        username, message_ids=message_id
    )
    assert not isinstance(pyrogram_message, list)

    if target_dir is None:
        target_dir = Path("./downloads/")

    if file_name is None:
        file_name = extract_file_name_from_pyrogram_message(
            pyrogram_message, use_original_file_name
        )

    file_path = target_dir / file_name
    file_path = file_path.absolute()

    result = await pyrogram_message.download(
        file_name=str(file_path), in_memory=in_memory
    )
    if in_memory:
        assert isinstance(result, BinaryIO)
        return result
    else:
        return file_path


async def download_file_via_subprocess(
    message_id,
    username,
    api_id: int,
    api_hash: str,
    bot_token: str,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
    in_memory: bool = False,
):
    """
    Download file using subprocess to avoid pyrogram/aiogram conflicts.
    Runs this same script as a standalone process.
    """
    # pyrogram doesn't work with aiogram, so we need to run this script as a separate process
    script_path = Path(__file__)
    assert bot_token is not None
    assert api_id is not None
    assert api_hash is not None

    # Build command arguments
    cmd = [
        "python",
        str(script_path),
        "--message_id",
        str(message_id),
        "--username",
        username,
        "--api_id",
        str(api_id),
        "--api_hash",
        api_hash,
        "--bot_token",
        bot_token,
    ]

    if target_dir is not None:
        cmd.extend(["--target_dir", str(target_dir)])

    if file_name is not None:
        cmd.extend(["--file_name", file_name])

    if use_original_file_name:
        cmd.append("--use_original_file_name")

    # Run subprocess asynchronously
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"Download subprocess failed: {error_msg}")

    script_output = stdout.decode()

    # Parse downloaded file path from script output with regex
    match = re.search(r"Downloaded file path: (.*)", script_output)
    if not match:
        raise RuntimeError(
            f"Could not parse downloaded file path from output: {script_output}"
        )

    downloaded_file_path = match.group(1).strip()

    if in_memory:
        # todo: check if this complies with BytesIO type annotation
        return open(downloaded_file_path, "rb")
    else:
        return Path(downloaded_file_path).absolute()


__all__ = [
    "download_file_from_aiogram_message",
    "download_file_via_subprocess",
    "download_file_with_pyrogram",
    "get_pyrogram_client",
    "get_media_and_media_type",
    "extract_file_name_from_pyrogram_message",
    "_check_aiogram_running",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download media from Telegram")

    parser.add_argument("--message_id", type=int, required=True)
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--target_dir", type=Path, required=False)
    parser.add_argument("--file_name", type=str, required=False)
    parser.add_argument("--use_original_file_name", action="store_true")
    parser.add_argument("--api_id", type=int, required=True)
    parser.add_argument("--api_hash", type=str, required=True)
    parser.add_argument("--bot_token", type=str, required=True)

    args = parser.parse_args()

    downloaded_file_path = asyncio.run(
        download_file_with_pyrogram(
            args.message_id,
            args.username,
            target_dir=args.target_dir,
            file_name=args.file_name,
            use_original_file_name=args.use_original_file_name,
            api_id=args.api_id,
            api_hash=args.api_hash,
            bot_token=args.bot_token,
            check_aiogram=False,  # Disable aiogram check in subprocess
            in_memory=False,
        )
    )
    assert isinstance(downloaded_file_path, Path)
    downloaded_file_path = downloaded_file_path.absolute()

    print(f"Downloaded file path: {downloaded_file_path}")
