# Done!
import pathlib
from typing import BinaryIO, Optional, Union

from aiogram.types import Message as AiogramMessage


async def download_file_aiogram(
    message: AiogramMessage,
    destination: Optional[Union[BinaryIO, pathlib.Path, str]] = None,
) -> Union[BinaryIO, pathlib.Path]:
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

    return result
