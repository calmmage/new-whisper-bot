from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Basic app configuration"""

    telegram_bot_token: SecretStr

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class AppBase(ABC):
    """Abstract base class defining whisper bot workflow interfaces"""

    # @abstractmethod
    # async def get_media_message(self, message_id: int, user_id: int):
    #     """Get media message from user"""
    #     pass

    @abstractmethod
    async def download_media(self, message_id: int, username: str) -> Path:
        """Download media from message"""
        pass

    @abstractmethod
    async def convert_video_to_audio(self, media_path: Path) -> Path:
        """Convert video to audio if necessary"""
        pass

    @abstractmethod
    async def cut_audio_into_pieces(self, audio_path: Path) -> list[Path]:
        """Cut audio into smaller pieces"""
        pass

    @abstractmethod
    async def parse_audio_chunks(self, audio_pieces: list[Path]) -> list[str]:
        """Parse audio pieces using LLM"""
        pass

    @abstractmethod
    async def merge_transcription_chunks(self, transcription_pieces: list[str]) -> str:
        """Merge transcription pieces back together"""
        pass

    # @abstractmethod
    # async def send_transcription_response(self, transcription: str, user_id: int):
    #     """Send transcription response to user"""
    #     pass

    @abstractmethod
    async def create_summary(self, transcription: str) -> str:
        """Create summary of transcription using LLM"""
        pass

    # @abstractmethod
    # async def send_summary_response(self, summary: str, user_id: int):
    #     """Send summary response to user"""
    #     pass

    async def run(self, message_id: int, username: str) -> str:
        """Main workflow that orchestrates all processing steps"""
        # Get media message from user
        # message = await self.get_media_message(message_id, user_id)

        # Download media
        media_path = await self.download_media(message_id, username)

        # Convert video to audio if necessary
        audio_path = await self.convert_video_to_audio(media_path)

        # Cut audio in pieces
        audio_pieces = await self.cut_audio_into_pieces(audio_path)

        # Parse using llm
        transcription_pieces = await self.parse_audio_chunks(audio_pieces)

        # Merge pieces back
        full_transcription = await self.merge_transcription_chunks(transcription_pieces)

        # Send response to user
        # await self.send_transcription_response(full_transcription, user_id)

        # # Create summary and send in separate message
        # summary = await self.create_summary(full_transcription)
        # await self.send_summary_response(summary, user_id)

        return full_transcription


class App(AppBase):
    name = "New Whisper Bot"

    def __init__(self, **kwargs):
        self.config = AppConfig(**kwargs)
