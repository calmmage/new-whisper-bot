from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings

from src.utils.convert_to_mp3 import convert_to_mp3
from src.utils.create_summary import create_summary
from src.utils.cut_audio import cut_audio_into_pieces
from src.utils.download_media import download_file_via_subprocess
from src.utils.merge_transcription_chunks import merge_transcription_chunks
from src.utils.parse_audio_chunks import parse_audio_chunks


class AppConfig(BaseSettings):
    """Basic app configuration"""

    telegram_bot_token: SecretStr
    telegram_api_id: int
    telegram_api_hash: SecretStr

    downloads_dir: Path = Path("downloads").absolute()
    cleanup_downloads: bool = True

    openai_api_key: Optional[SecretStr] = None
    use_memory_profiler: bool = False

    # todo: delete unused fields
    # Whisper API and audio chunking settings
    # whisper_chunk_duration: int = 600  # 10 minutes in seconds
    whisper_chunk_duration: int = 120  # 2 minutes in seconds
    # whisper_overlap_duration: int = 30  # 30 seconds
    whisper_overlap_duration: int = 5  # 30 seconds
    # actually, my account says 7500 rpm
    # todo: disentangle the cutting logic from rate limit, set target chunk amount instead.
    whisper_rate_limit: int = 50  # Maximum number of chunks to create
    # not sure - maybe 100? This is a global limit so need to be careful to not shoot me in the leg.
    # todo: double-check
    whisper_max_concurrent: int = 50  # Maximum number of concurrent API calls
    # todo: file size limit is 25 megabytes - look out for that instead..

    target_chunk_count: int = 20
    max_chunk_size: int = 20 * 60  # 20 min
    min_chunk_size: int = 2 * 60  # 2 min

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# todo: use this.
def calculate_optimal_chunk_size(
    audio_size,
    target_chunk_count=20,
    max_chunk_size=20 * 60,  # 20 min,
    min_chunk_size=2 * 60,  # 2 min,
):
    target_chunk_size = int(audio_size / target_chunk_count)
    if target_chunk_size < min_chunk_size:
        return min_chunk_size
    if target_chunk_size > max_chunk_size:
        return max_chunk_size
    return target_chunk_size


class AppBase(ABC):
    """Abstract base class defining whisper bot workflow interfaces"""

    @abstractmethod
    async def download_media(self, message_id: int, username: str) -> Path:
        """Download media from message"""
        pass

    @abstractmethod
    async def convert_video_to_audio(self, media_path: Path) -> Path:
        """Convert video to audio if necessary"""
        pass

    @abstractmethod
    async def cut_audio_into_pieces(self, audio_path: Path) -> List[Path]:
        """Cut audio into smaller pieces"""
        pass

    @abstractmethod
    async def parse_audio_chunks(self, audio_pieces: List[Path]) -> List[str]:
        """Parse audio pieces using LLM"""
        pass

    @abstractmethod
    async def merge_transcription_chunks(
        self, transcription_pieces: List[str], username: Optional[str] = None
    ) -> str:
        """Merge transcription pieces back together"""
        pass

    @abstractmethod
    async def create_summary(
        self, transcription: str, username: Optional[str] = None
    ) -> str:
        """Create summary of transcription using LLM"""
        pass

    async def run(self, message_id: int, username: str, model="whisper-1") -> str:
        """Main workflow that orchestrates all processing steps"""
        # Download media
        media_path = await self.download_media(message_id, username)

        # Convert video to audio if necessary
        audio_path = await self.convert_video_to_audio(media_path)

        # Cut audio in pieces
        audio_pieces = await self.cut_audio_into_pieces(audio_path)

        # Parse using OpenAI Whisper API
        transcription_pieces = await self.parse_audio_chunks(audio_pieces, model=model)

        # Merge pieces back
        full_transcription = await self.merge_transcription_chunks(
            transcription_pieces, username
        )

        return full_transcription


class App(AppBase):
    name = "New Whisper Bot"

    def __init__(self, **kwargs):
        self.config = AppConfig(**kwargs)

    async def download_media(self, message_id: int, username: str) -> Path:
        """Download media from message using aiogram"""
        # Use subprocess to avoid aiogram/pyrogram conflicts
        return await download_file_via_subprocess(
            message_id=message_id,
            username=username,
            bot_token=self.config.telegram_bot_token.get_secret_value(),
            api_id=self.config.telegram_api_id,
            api_hash=self.config.telegram_api_hash.get_secret_value(),
            target_dir=self.config.downloads_dir,
        )

    async def convert_video_to_audio(self, media_path: Path) -> Path:
        """Convert video to audio if necessary using ffmpeg"""
        path = await convert_to_mp3(
            source_path=media_path, use_memory_profiler=self.config.use_memory_profiler
        )
        # cleanup - delete original file if conversion was successful
        if self.config.cleanup_downloads and not media_path.name.endswith(".mp3"):
            media_path.unlink()
        return path

    async def cut_audio_into_pieces(self, audio_path: Path) -> List[Path]:
        """Cut audio into smaller pieces using ffmpeg"""
        chunks = await cut_audio_into_pieces(
            audio_path=audio_path,
            chunk_duration=self.config.whisper_chunk_duration,
            overlap_duration=self.config.whisper_overlap_duration,
            rate_limit=self.config.whisper_rate_limit,
            use_memory_profiler=self.config.use_memory_profiler,
        )

        # cleanup - delete original file if cutting was successful
        if self.config.cleanup_downloads:
            audio_path.unlink()

        return chunks

    async def parse_audio_chunks(
        self, audio_pieces: List[Path], model="whisper-1"
    ) -> List[str]:
        """Parse audio pieces using OpenAI Whisper API"""
        api_key = None
        if self.config.openai_api_key:
            api_key = self.config.openai_api_key.get_secret_value()

        transcribed_chunks = await parse_audio_chunks(
            audio_chunks=audio_pieces,
            model_name=model,
            api_key=api_key,
            max_concurrent=self.config.whisper_max_concurrent,
        )

        # cleanup - delete audio pieces if parsing was successful
        if self.config.cleanup_downloads:
            for chunk in audio_pieces:
                chunk.unlink()

        return transcribed_chunks

    async def merge_transcription_chunks(
        self, transcription_pieces: List[str], username: Optional[str] = None
    ) -> str:
        """Merge transcription pieces back together using custom text_utils"""
        return await merge_transcription_chunks(
            transcription_chunks=transcription_pieces,
            buffer=25,
            match_cutoff=15,
            username=username,
        )

    async def create_summary(
        self, transcription: str, username: Optional[str] = None
    ) -> str:
        """Create summary of transcription using botspot's llm_provider"""
        return await create_summary(
            transcription=transcription,
            username=username,
        )
