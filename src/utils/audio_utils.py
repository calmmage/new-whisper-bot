from io import BytesIO
from typing import BinaryIO
from loguru import logger
from typing import List, Union
from pydub import AudioSegment

DEFAULT_PERIOD = 120 * 1000  # 2 minutes
DEFAULT_BUFFER = 5 * 1000  # 5 seconds

Audio = Union[AudioSegment, BytesIO, BinaryIO, str]

def split_audio(
    audio: Audio, period=DEFAULT_PERIOD, buffer=DEFAULT_BUFFER
) -> List[BytesIO]:
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
