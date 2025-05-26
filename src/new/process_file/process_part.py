from pathlib import Path


async def process_part(part) -> str:
    """
    process file using pydub fully in-memory.

    THIS IS ALREADY IMPLEMENTED IN OLD WHISPER BOT - COPY IT OVER!
    """

    if isinstance(part, Path):
        # file on disk - load.
        raise NotImplementedError
    else:
        # in-memory Audio.
        raise NotImplementedError

    return result
