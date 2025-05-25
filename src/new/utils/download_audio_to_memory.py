from aiogram.types import Message as AiogramMessage


async def download_media_to_memory(message: AiogramMessage) -> bytes:
    """
    Download audio file to memory and return its content as bytes.

    :param audio_file: The audio file content in bytes.
    :param file_name: The name of the file (not used in this function).
    :return: The audio file content as bytes.
    """
    # In this case, we assume audio_file is already in bytes format
    # call the pyrogram downloader with right parameters

    return media_file
