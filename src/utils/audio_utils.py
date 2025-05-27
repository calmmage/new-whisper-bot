from io import BytesIO
from typing import BinaryIO
from loguru import logger
from typing import Sequence, Union
from pydub import AudioSegment
from pathlib import Path

DEFAULT_PERIOD = 120 * 1000  # 2 minutes
DEFAULT_BUFFER = 5 * 1000  # 5 seconds

Audio = Union[AudioSegment, BytesIO, BinaryIO, Path, str]


def split_audio(
    audio: Audio, period=DEFAULT_PERIOD, buffer=DEFAULT_BUFFER, return_as_files: bool = True
) -> Sequence[Union[AudioSegment, BytesIO]]:
    """
    Splits an audio input into smaller chunks based on the specified duration period and
    optional buffer overlap. The chunks can either be returned as a list of audio segments
    or as in-memory audio files in MP3 format.

    Parameters:
        audio (Audio): The input audio data that can be a file path (str),
                       a file-like object (BytesIO or BinaryIO), or an AudioSegment.
        period: The duration (in milliseconds) of each audio chunk.
        buffer: The overlap (in milliseconds) between consecutive chunks.
        return_as_files (bool): If True, the chunks will be returned as
                                in-memory MP3 files. Otherwise, they will
                                be returned as AudioSegment objects.

    Returns:
        Sequence[Union[AudioSegment, BytesIO]]: A list of the audio chunks, where each
                                                chunk is either an AudioSegment or an
                                                in-memory audio file (BytesIO) if
                                                `return_as_files` is True.
    """
    if isinstance(audio, (str, BytesIO, BinaryIO)):
        logger.debug(f"Loading audio from {audio}")
        audio = AudioSegment.from_file(audio)

    assert isinstance(audio, AudioSegment)

    chunks = []
    s = 0

    logger.debug("Splitting audio into chunks")
    while s + period < len(audio):
        chunks.append(audio[s : s + period])
        s += period - buffer
    chunks.append(audio[s:])
    logger.debug(f"Split into {len(chunks)} chunks")

    if not return_as_files:
        return chunks

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
