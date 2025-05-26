from io import BytesIO
from typing import BinaryIO

from typing import List
import loguru
from bot_base.utils.gpt_utils import (
    WHISPER_RATE_LIMIT,
    Audio,
)
from pydub import AudioSegment

DEFAULT_PERIOD = 120 * 1000  # 2 minutes
DEFAULT_BUFFER = 5 * 1000  # 5 seconds


def split_audio(
    audio: Audio, period=DEFAULT_PERIOD, buffer=DEFAULT_BUFFER, logger=None
) -> List[BytesIO]:
    if isinstance(audio, (str, BytesIO, BinaryIO)):
        logger.debug(f"Loading audio from {audio}")
        audio = AudioSegment.from_file(audio)
    if logger is None:
        logger = loguru.logger
    chunks = []
    s = 0

    if len(audio) / period > WHISPER_RATE_LIMIT - 5:
        period = len(audio) // (WHISPER_RATE_LIMIT - 5)

    logger.debug("Splitting audio into chunks")
    while s + period < len(audio):
        chunks.append(audio[s : s + period])
        s += period - buffer
    chunks.append(audio[s:])
    logger.debug(f"Split into {len(chunks)} chunks")

    in_memory_audio_files = []

    logger.debug("Converting chunks to mp3")
    for i, chunk in enumerate(chunks):
        buffer = BytesIO()
        chunk.export(buffer, format="mp3")  # check which format it is and
        # use the same
        buffer.name = f"chunk_{i}.mp3"
        in_memory_audio_files.append(buffer)
    logger.debug("Converted chunks to mp3")

    return in_memory_audio_files
