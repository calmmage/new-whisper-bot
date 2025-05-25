"""
Dynamically decide which way to download a file

Option 1 - with aiogram

Option 2 - with pyrogram

"""

from aiogram.types import Message as AiogramMessage


def download_file(
    message: AiogramMessage,
    # todo: accept a parameter to download to memory. the same way as aiogram accepts it
):
    try:
        # use my botspot aiogram utils for downloading
        from botspot.utils.unsorted import _download_telegram_file_aiogram

        result = _download_telegram_file_aiogram()

    except:
        # use pyrogram downloader

        from src.core.download_media import download_file_via_subprocess

        result = download_file_via_subprocess()
