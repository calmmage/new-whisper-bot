from aiogram.types import Message as AiogramMessage
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Union, BinaryIO, List
import asyncio

from src.new.utils.convert_to_mp3_ffmpeg import convert_to_mp3
from src.new.utils.download_attachment import download_file
from src.new.utils.format_text_with_llm import format_text_with_llm
from src.new.utils.text_utils import merge_all_chunks


class AppConfig(BaseSettings):
    telegram_api_id: int
    telegram_api_hash: SecretStr
    telegram_bot_token: SecretStr

    downloads_dir: Path = Path("downloads")
    use_original_file_name: bool = False

    openai_max_concurrent_connections: int = 50

    # todo: use
    transcription_model: str = "whisper-1"  # Default transcription model
    summary_model: str = "claude-4-sonnet"
    # todo: use
    chat_model: str = (
        "claude-4-sonnet"  # Default chat model for discussing transcripts and summaries
    )
    # todo: use
    formatting_model: str = "gpt-4.1-nano"


class App:
    def __init__(self, **kwargs):
        self.config = AppConfig(**kwargs)
        self.semaphore = asyncio.Semaphore(
            self.config.openai_max_concurrent_connections
        )

    async def process_message(self, message: AiogramMessage):
        """
        [x] process_message
        ├── [x] download_attachment
        ├── prepare_parts
        │   ├── [x] convert_to_mp3_ffmpeg (if disk)
        │   ├── cut_audio_inplace_ffmpeg (if disk)
        ├── process_parts
        │   ├── convert_to_mp3_pydub
        │   ├── cut_audio_inmemory_pydub
        │   └── transcribe_parallel
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

    async def process_file_on_disk(self, media_file: Path) -> list:
        if media_file.suffix != ".mp3":
            mp3_file = await convert_to_mp3(media_file)
            # delete original file
            media_file.unlink()
            media_file = mp3_file

        # todo: cut audio inplace with ffmpeg

        # TODO: implement disk-based processing
        # │   ├── convert_to_mp3_ffmpeg (if disk)
        # │   ├── cut_audio_inplace_ffmpeg (if disk)
        return []

    # endregion prepare_parts

    # region process_parts
    async def process_parts(self, parts: list) -> list:
        chunks = []

        # this has to be done one by one, do NOT parallelize.
        for part in parts:
            # todo: make sure memory is freed after each part.
            chunks += await self.process_part(part)

        return chunks

    async def process_part(self, part: Union[BinaryIO, Path]) -> list:
        """
        find and use the exact flow from old whisper bot?
        """
        if isinstance(part, Path):
            # load with pydub
            raise NotImplementedError
        else:
            # load with pydub from binary io
            raise NotImplementedError

        # step 1
        # │   ├── convert_to_mp3_pydub
        # │   ├── cut_audio_inmemory_pydub
        # │   └── transcribe_parallel
        # TODO: implement part processing
        return []

    # endregion process_parts

    # region format_chunks_with_llm

    async def format_chunks_with_llm(self, chunks: List[str]) -> List[str]:
        # Create tasks for all chunks
        tasks = [self.process_chunk(chunk) for chunk in chunks]

        # todo: calculate and log average and total time taken to format chunks.
        # Wait for all tasks to complete
        formatted_chunks = await asyncio.gather(*tasks)

        return formatted_chunks

    async def process_chunk(self, chunk: str) -> str:
        async with self.semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)
            return await format_text_with_llm(
                text=chunk,
                model=self.config.formatting_model,
            )

    # endregion format_chunks_with_llm
