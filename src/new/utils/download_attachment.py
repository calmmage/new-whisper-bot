from typing import BinaryIO, Optional, Union
from pathlib import Path
from loguru import logger
from aiogram.types import Message as AiogramMessage

from src.new.utils.download_attachment_aiogram import download_file_aiogram


async def download_file(
    message: AiogramMessage,
    target_dir: Optional[Path] = None,
    file_name: Optional[str] = None,
    use_original_file_name: bool = True,
    api_id: Optional[int] = None,
    api_hash: Optional[str] = None,
    bot_token: Optional[str] = None,
    in_memory: Optional[bool] = None,
) -> Union[BinaryIO, Path]:
    try:
        # use my botspot aiogram utils for downloading
        return await download_file_aiogram(
            message=message,
            target_dir=target_dir,
            file_name=file_name,
            use_original_file_name=use_original_file_name,
            in_memory=True if in_memory is None else in_memory,
        )

    except Exception as e:  # todo: specify api error
        # use pyrogram downloader
        logger.warning(
            f"Generic error captured, update code to handle specific error, Type: {type(e)}, Error: {e}"
        )
        import traceback

        traceback.print_exc()

        from src.new.utils.download_attachment_pyrogram import (
            download_file_from_aiogram_message,
        )

        result = await download_file_from_aiogram_message(
            message=message,
            target_dir=target_dir,
            file_name=file_name,
            use_original_file_name=use_original_file_name,
            api_id=api_id,
            api_hash=api_hash,
            bot_token=bot_token,
            in_memory=False if in_memory is None else in_memory,
            use_subprocess=True,
        )

        return result
