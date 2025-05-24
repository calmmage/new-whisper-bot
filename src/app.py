import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings

from src.utils.convert_video_to_audio import convert_video_to_audio
from src.utils.cut_audio import cut_audio_into_pieces
from src.utils.download_media import download_file_from_aiogram_message
from src.utils.merge_transcription_chunks import merge_transcription_chunks
from src.utils.parse_audio_chunks import parse_audio_chunks
from src.utils.create_summary import create_summary


class AppConfig(BaseSettings):
    """Basic app configuration"""

    telegram_bot_token: SecretStr
    openai_api_key: Optional[SecretStr] = None
    use_memory_profiler: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


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
    async def merge_transcription_chunks(self, transcription_pieces: List[str], username: Optional[str] = None) -> str:
        """Merge transcription pieces back together"""
        pass

    @abstractmethod
    async def create_summary(self, transcription: str, username: Optional[str] = None) -> str:
        """Create summary of transcription using LLM"""
        pass

    async def run(self, message_id: int, username: str) -> str:
        """Main workflow that orchestrates all processing steps"""
        # Download media
        media_path = await self.download_media(message_id, username)

        # Convert video to audio if necessary
        audio_path = await self.convert_video_to_audio(media_path)

        # Cut audio in pieces
        audio_pieces = await self.cut_audio_into_pieces(audio_path)

        # Parse using OpenAI Whisper API
        transcription_pieces = await self.parse_audio_chunks(audio_pieces)

        # Merge pieces back
        full_transcription = await self.merge_transcription_chunks(transcription_pieces, username)

        return full_transcription


class App(AppBase):
    name = "New Whisper Bot"

    def __init__(self, **kwargs):
        self.config = AppConfig(**kwargs)

    async def download_media(self, message_id: int, username: str) -> Path:
        """Download media from message using aiogram"""
        # Use subprocess to avoid aiogram/pyrogram conflicts
        return await download_file_from_aiogram_message(
            message_id=message_id,
            username=username,
            use_subprocess=True
        )

    async def convert_video_to_audio(self, media_path: Path) -> Path:
        """Convert video to audio if necessary using ffmpeg"""
        return await convert_video_to_audio(
            video_path=media_path,
            use_memory_profiler=self.config.use_memory_profiler
        )

    async def cut_audio_into_pieces(self, audio_path: Path) -> List[Path]:
        """Cut audio into smaller pieces using ffmpeg"""
        return await cut_audio_into_pieces(
            audio_path=audio_path,
            chunk_duration=600,  # 10 minutes
            overlap_duration=30,  # 30 seconds
            use_memory_profiler=self.config.use_memory_profiler
        )

    async def parse_audio_chunks(self, audio_pieces: List[Path]) -> List[str]:
        """Parse audio pieces using OpenAI Whisper API"""
        api_key = None
        if self.config.openai_api_key:
            api_key = self.config.openai_api_key.get_secret_value()

        return await parse_audio_chunks(
            audio_chunks=audio_pieces,
            model_name="whisper-1",
            api_key=api_key,
            max_concurrent=3  # Limit concurrent API calls
        )

    async def merge_transcription_chunks(self, transcription_pieces: List[str], username: Optional[str] = None) -> str:
        """Merge transcription pieces back together using custom text_utils"""
        return await merge_transcription_chunks(
            transcription_chunks=transcription_pieces,
            buffer=25,
            match_cutoff=15,
            username=username
        )

    async def create_summary(self, transcription: str, username: Optional[str] = None) -> str:
        """Create summary of transcription using botspot's llm_provider"""
        return await create_summary(
            transcription=transcription,
            max_length=1000,
            username=username,
            model="gpt-4-1106-preview"  # Use GPT-4 Turbo for better summaries
        )
