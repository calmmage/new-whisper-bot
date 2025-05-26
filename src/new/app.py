import asyncio
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import openai
from aiogram.types import Message as AiogramMessage
from loguru import logger
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydub import AudioSegment
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.new.create_summary import create_summary
from src.new.utils.audio_utils import split_audio
from src.new.utils.convert_to_mp3_ffmpeg import convert_to_mp3_ffmpeg
from src.new.utils.cut_audio_ffmpeg import cut_audio_ffmpeg
from src.new.utils.download_attachment import download_file
from src.new.utils.format_text_with_llm import format_text_with_llm
from src.new.utils.text_utils import merge_all_chunks


class AppConfig(BaseSettings):
    telegram_api_id: int
    telegram_api_hash: SecretStr
    telegram_bot_token: SecretStr

    downloads_dir: Path = Path("downloads")
    use_original_file_name: bool = False

    # cutting parameters
    target_part_size: int = 25 * 1024 * 1024  # desired max size in bytes
    target_chunk_count = 20
    minimum_chunk_duration = 2 * 60 * 1000  # 2 minutes
    maximum_chunk_duration = 20 * 60 * 1000  # 20 minutes
    overlap_duration = 5 * 1000  # # 5 seconds

    # todo: decide global semaphore or per task type?
    # openai_max_concurrent_connections: int = 50
    whisper_max_concurrent: int = 50
    summary_max_concurrent: int = 10
    format_max_concurrent: int = 50
    chat_max_concurrent: int = 10

    openai_api_key: SecretStr

    # todo: allow user to configure
    transcription_model: str = "whisper-1"  # Default transcription model
    # todo: allow user to configure
    summary_model: str = "claude-4-sonnet"
    summary_max_tokens: int = 2048  # Default max tokens for summary
    # todo: use
    # todo: allow user to configure
    chat_model: str = (
        "claude-4-sonnet"  # Default chat model for discussing transcripts and summaries
    )
    # todo: use
    # todo: allow user to configure
    formatting_model: str = "gpt-4.1-nano"


class App:
    def __init__(self, **kwargs):
        self.config = AppConfig(**kwargs)
        # self.semaphore = asyncio.Semaphore(
        #     self.config.openai_max_concurrent_connections
        # )
        # todo: use
        self.whisper_semaphore = asyncio.Semaphore(self.config.whisper_max_concurrent)
        # todo: use
        self.summary_semaphore = asyncio.Semaphore(self.config.summary_max_concurrent)
        self.format_semaphore = asyncio.Semaphore(self.config.format_max_concurrent)
        # todo: use
        self.chat_semaphore = asyncio.Semaphore(self.config.chat_max_concurrent)

        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key.get_secret_value()
        )

    async def process_message(self, message: AiogramMessage):
        """
        [x] process_message
        ├── [x] download_attachment
        ├── prepare_parts
        │   ├── [x] convert_to_mp3_ffmpeg (if disk)
        │   ├── [x] cut_audio_inplace_ffmpeg (if disk)
        ├── [x] process_parts
        │   ├── [x] convert_to_mp3_pydub
        │   ├── [x] cut_audio_inmemory_pydub
        │   └── [x] transcribe_parallel
        ├── [x] format_chunks_with_llm
        └── [x] merge_chunks

        Rewrite old code as a comprehensive util!!! with clear explicit params
        │   ├── cut_audio_inplace_ffmpeg (if disk)
        """
        # in_memory_audio = download_audio_to_memory(message)
        media_file = await self.download_attachment(message)

        parts = await self.prepare_parts(media_file)

        chunks = await self.process_parts(parts)

        chunks = await self.format_chunks_with_llm(chunks)

        result = merge_all_chunks(chunks)

        return result

        # audio_file_size = calculate_audio_size(in_memory_audio)
        # AUDIO_SIZE_THRESHOLD = 100 * 1024 * 1024  # 50 MB
        # offload files over 50 megabytes to disk

    async def download_attachment(
        self, message: AiogramMessage
    ) -> Union[BinaryIO, Path]:
        return await download_file(
            message=message,
            target_dir=self.config.downloads_dir,
            # file_name=file_name,
            use_original_file_name=self.config.use_original_file_name,
            api_id=self.config.telegram_api_id,
            api_hash=self.config.telegram_api_hash.get_secret_value(),
            bot_token=self.config.telegram_bot_token.get_secret_value(),
            in_memory=None,
        )

    # region prepare_parts
    async def prepare_parts(
        self, media_file: Union[BinaryIO, Path]
    ) -> List[Union[BinaryIO, Path]]:
        if isinstance(media_file, Path):
            # process file on disk - with
            return await self.process_file_on_disk(media_file)
        else:
            # process file in memory - with pydub
            assert isinstance(media_file, BinaryIO)
            return [media_file]

    async def process_file_on_disk(self, media_file: Path) -> List[Path]:
        if media_file.suffix != ".mp3":
            mp3_file = await convert_to_mp3_ffmpeg(media_file)
            # delete original file
            media_file.unlink()
            media_file = mp3_file

        # todo: cut audio inplace with ffmpeg
        parts = await self.cut_audio_with_ffmpeg(media_file)

        return parts

    async def cut_audio_with_ffmpeg(self, media_file):
        file_size = self._get_file_size(media_file)
        num_parts = int(file_size / self.config.target_part_size) + 1

        logger.info(
            f"Cutting file sized {file_size / (1024 * 1024):.2f} MB into {num_parts} parts - target size {self.config.target_part_size / (1024 * 1024):.2f} MB"
        )
        if num_parts > 1:
            parts = await cut_audio_ffmpeg(media_file, num_parts=num_parts)
            media_file.unlink()
            return parts
        else:
            return [media_file]

    @staticmethod
    def _get_file_size(media_file):
        return media_file.stat().st_size

    # endregion prepare_parts

    # region process_parts
    async def process_parts(self, parts: list) -> list:
        chunks = []

        # this has to be done one by one, do NOT parallelize.
        for part in parts:
            # todo: make sure memory is freed after each part.
            chunks += await self.process_part(part)

        return chunks

    @staticmethod
    def _determine_optimal_period(
        audio_size,
        target_chunk_count=20 * 1000,
        max_chunk_size=20 * 60 * 1000,  # 20 min,
        min_chunk_size=2 * 60 * 1000,  # 2 min,
    ):
        target_chunk_size = int(audio_size / target_chunk_count)
        if target_chunk_size < min_chunk_size:
            return min_chunk_size
        if target_chunk_size > max_chunk_size:
            return max_chunk_size
        return target_chunk_size

    async def process_part(
        self,
        audio: Union[BinaryIO, Path],
        # period: int = DEFAULT_PERIOD, # - use
        # buffer: int = DEFAULT_BUFFER,
    ) -> List[str]:
        """
        find and use the exact flow from old whisper bot?

        # │   ├── convert_to_mp3_pydub
        # │   ├── cut_audio_inmemory_pydub
        # │   └── transcribe_parallel
        """

        if isinstance(audio, (str, BytesIO, BinaryIO)):
            logger.debug(f"Loading audio from {audio}")
            audio = AudioSegment.from_file(audio)

        audio_duration = len(audio)
        period = self._determine_optimal_period(
            audio_duration,
            target_chunk_count=self.config.target_chunk_count,
            max_chunk_size=self.config.maximum_chunk_duration,
            min_chunk_size=self.config.minimum_chunk_duration,
        )

        audio_chunks = split_audio(
            audio, period=period, buffer=self.config.overlap_duration, logger=logger
        )

        # todo: add nice logging - here and everywhere..
        transcriptions = await self.parse_audio_chunks(audio_chunks)
        # todo: report user on our progress - update status message.

        return transcriptions

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        reraise=True,
    )
    async def parse_audio_chunk(
        self,
        audio_file: BytesIO,
        model_name: str = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Parse a single audio chunk using OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file
            model_name: Name of the Whisper model to use (whisper-1)
            language: Language code (optional, auto-detected if None)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)

        Returns:
            Transcription text
        """
        if model_name is None:
            model_name = self.config.transcription_model

        logger.info(f"Transcribing {audio_file.name} using OpenAI Whisper API")

        async with self.whisper_semaphore:
            # with open(audio_path, "rb") as audio_file:
            # Call the OpenAI API
            options = {"response_format": "text"}
            if language:
                options["language"] = language

            response = await self.openai_client.audio.transcriptions.create(
                file=audio_file, model=model_name, **options
            )

            # Extract the transcription text
            transcription = response
            logger.info(
                f"Transcription complete for {audio_file.name}: {len(transcription)} characters"
            )

            return transcription

    async def parse_audio_chunks(
        self,
        audio_chunks: List[BytesIO],
        model_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[str]:
        """
        Parse multiple audio chunks using OpenAI Whisper API.

        Args:
            audio_chunks: List of paths to audio files
            model_name: Name of the Whisper model to use (whisper-1)
            language: Language code (optional, auto-detected if None)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            max_concurrent: Maximum number of concurrent transcriptions

        Returns:
            List of transcription texts
        """
        logger.info(f"Transcribing {len(audio_chunks)} audio chunks")

        # Create tasks for all chunks
        # todo: calculate and log average and total time taken to format chunks.
        tasks = [
            self.parse_audio_chunk(chunk, model_name, language=language)
            for chunk in audio_chunks
        ]

        # Wait for all tasks to complete
        transcriptions = await asyncio.gather(*tasks)

        logger.info(f"Transcription complete for all {len(audio_chunks)} chunks")
        return transcriptions

    # endregion process_parts

    # region format_chunks_with_llm

    async def format_chunks_with_llm(self, chunks: List[str]) -> List[str]:
        # Create tasks for all chunks
        tasks = [self.format_chunk(chunk) for chunk in chunks]

        # todo: calculate and log average and total time taken to format chunks.
        # Wait for all tasks to complete
        formatted_chunks = await asyncio.gather(*tasks)

        return formatted_chunks

    async def format_chunk(self, chunk: str) -> str:
        async with self.format_semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)
            return await format_text_with_llm(
                text=chunk,
                model=self.config.formatting_model,
            )

    # endregion format_chunks_with_llm

    async def create_summary(self, transcript: str, username: str):
        """
        Create a summary of the transcript using the configured model.
        """
        async with self.summary_semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)
            return await create_summary(
                transcription=transcript,
                username=username,
                model=self.config.summary_model,
                max_tokens=self.config.summary_max_tokens,
            )
