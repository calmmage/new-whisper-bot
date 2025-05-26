import asyncio
import datetime
import time
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union, Sequence, Any

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

from botspot.components.new.llm_provider import aquery_llm_text
from botspot.components.data.mongo_database import get_database
from botspot.core.errors import BotspotError
from src.utils.create_summary import create_summary
from src.utils.audio_utils import split_audio, Audio
from src.utils.convert_to_mp3_ffmpeg import convert_to_mp3_ffmpeg
from src.utils.cut_audio_ffmpeg import cut_audio_ffmpeg
from src.utils.download_attachment import download_file
from src.utils.format_text_with_llm import format_text_with_llm
from src.utils.text_utils import merge_all_chunks


class AppConfig(BaseSettings):
    telegram_api_id: int
    telegram_api_hash: SecretStr
    telegram_bot_token: SecretStr

    downloads_dir: Path = Path("downloads").absolute()
    # todo: make sure that if there's only a single file - we reuse it and not delete it
    cleanup_downloads: bool = True
    use_original_file_name: bool = False

    # cutting parameters
    target_part_size: int = 25 * 1024 * 1024  # desired max size in bytes
    target_chunk_count: int = 20
    minimum_chunk_duration: int = 2 * 60 * 1000  # 2 minutes
    maximum_chunk_duration: int = 20 * 60 * 1000  # 20 minutes
    overlap_duration: int = 5 * 1000  # # 5 seconds

    # todo: check if we need a global semaphore instead
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
    chat_max_tokens: int = 2048
    # todo: allow user to configure
    formatting_model: str = "gpt-4.1-nano"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class App:
    name = "New Whisper Bot"

    def __init__(self, **kwargs):
        self.config = AppConfig(**kwargs)
        # self.semaphore = asyncio.Semaphore(
        #     self.config.openai_max_concurrent_connections
        # )
        self.whisper_semaphore = asyncio.Semaphore(self.config.whisper_max_concurrent)
        self.summary_semaphore = asyncio.Semaphore(self.config.summary_max_concurrent)
        self.format_semaphore = asyncio.Semaphore(self.config.format_max_concurrent)
        self.chat_semaphore = asyncio.Semaphore(self.config.chat_max_concurrent)

        self.openai_client = openai.AsyncOpenAI(
            api_key=self.config.openai_api_key.get_secret_value()
        )

        # Initialize MongoDB connection
        self._db = None

    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db



    async def log_event(self, event_type: str, user_id: str, data: Dict[str, Any] = None) -> None:
        """
        Log an event to MongoDB.

        Args:
            event_type: Type of event (file_submission, file_processing, chat)
            user_id: User ID or username
            data: Additional data to log
        """
        if data is None:
            data = {}

        event = {
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.datetime.utcnow(),
            **data
        }

        logger.info(f"Logging event: {event_type} for user {user_id}")
        await self.db.events.insert_one(event)

    async def log_cost(self, operation: str, user_id: str, model: str, cost: float = None, 
                      usage: Dict[str, Any] = None, message_id: int = None) -> None:
        """
        Log cost information to MongoDB.

        Args:
            operation: Type of operation (transcription, formatting, summary, chat)
            user_id: User ID or username
            model: Model used
            cost: Estimated cost (if available)
            usage: Usage information (tokens, etc.)
            message_id: Message ID for grouping costs by conversation
        """
        if usage is None:
            usage = {}

        cost_data = {
            "operation": operation,
            "user_id": user_id,
            "model": model,
            "timestamp": datetime.datetime.utcnow(),
            "usage": usage
        }

        if cost is not None:
            cost_data["cost"] = cost

        if message_id is not None:
            cost_data["message_id"] = message_id

        logger.info(f"Logging cost for {operation} using {model} for user {user_id}")
        await self.db.costs.insert_one(cost_data)

    async def get_total_cost(self, user_id: str, message_id: int = None) -> Dict[str, Any]:
        """
        Get the total cost for a specific user and optionally for a specific message.

        Args:
            user_id: User ID or username
            message_id: Message ID for filtering costs by conversation

        Returns:
            Dictionary with total cost and breakdown by operation
        """
        query = {"user_id": user_id}
        if message_id is not None:
            query["message_id"] = message_id

        costs = await self.db.costs.find(query).to_list(length=1000)

        total_cost = 0.0
        operation_costs = {}
        model_costs = {}

        for cost_entry in costs:
            cost = cost_entry.get("cost", 0.0)
            operation = cost_entry.get("operation", "unknown")
            model = cost_entry.get("model", "unknown")

            total_cost += cost

            # Track costs by operation
            if operation not in operation_costs:
                operation_costs[operation] = 0.0
            operation_costs[operation] += cost

            # Track costs by model
            if model not in model_costs:
                model_costs[model] = 0.0
            model_costs[model] += cost

        return {
            "total_cost": total_cost,
            "operation_costs": operation_costs,
            "model_costs": model_costs,
            "currency": "USD",
            "cost_count": len(costs)
        }

    # Main method
    async def process_message(self, message: AiogramMessage, whisper_model: Optional[str] = None) -> str:
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
        assert message.from_user is not None
        username = message.from_user.username
        message_id = message.message_id

        if username is None:
            raise BotspotError(
                f"User username is None for user {message.from_user.id}.",
                user_message="Your telegram account must have a @username set to check that you have access to the bot.",
            )

        # Log file submission event
        start_time = time.time()
        file_info = {
            "message_id": message_id,
            "chat_id": message.chat.id,
            "file_type": self._get_file_type(message)
        }
        logger.info(f"User {username} submitted a file for processing: {file_info}")
        await self.log_event("file_submission", username, file_info)

        media_file = await self.download_attachment(message)

        parts = await self.prepare_parts(media_file)

        # Store message_id in a context variable to pass to nested methods
        self._current_message_id = message_id

        chunks = await self.process_parts(parts, username=username, whisper_model=whisper_model)

        chunks = await self.format_chunks_with_llm(chunks, username=username)

        result = merge_all_chunks(chunks)

        # Log file processing completion
        processing_time = time.time() - start_time
        completion_info = {
            "message_id": message_id,
            "processing_time_seconds": processing_time,
            "result_length": len(result),
            "chunks_count": len(chunks)
        }
        logger.info(f"Completed processing file for user {username} in {processing_time:.2f} seconds")
        await self.log_event("file_processing_complete", username, completion_info)

        # Clear the context variable
        self._current_message_id = None

        return result

    def _get_file_type(self, message: AiogramMessage) -> str:
        """Determine the type of file in the message."""
        if message.audio:
            return "audio"
        elif message.voice:
            return "voice"
        elif message.video:
            return "video"
        elif message.document:
            return "document"
        elif message.video_note:
            return "video_note"
        else:
            return "unknown"

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
    async def prepare_parts(self, media_file: Union[BinaryIO, Path]) -> Sequence[Audio]:
        if isinstance(media_file, Path):
            # process file on disk - with
            parts = await self.process_file_on_disk(media_file)
            if self.config.cleanup_downloads:
                if media_file != parts[0]:
                    media_file.unlink(missing_ok=True)
            return parts
        else:
            # process file in memory - with pydub
            assert isinstance(media_file, (BinaryIO, BytesIO))
            return [media_file]

    async def process_file_on_disk(self, media_file: Path) -> List[Path]:
        if media_file.suffix != ".mp3":
            mp3_file = await convert_to_mp3_ffmpeg(media_file)
            # delete original file
            if self.config.cleanup_downloads:
                media_file.unlink(missing_ok=True)
            media_file = mp3_file

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
            if self.config.cleanup_downloads:
                media_file.unlink(missing_ok=True)
            return parts
        else:
            return [media_file]

    @staticmethod
    def _get_file_size(media_file):
        return media_file.stat().st_size

    # endregion prepare_parts

    # region process_parts
    async def process_parts(self, parts: Sequence[Audio], username: Optional[str] = None, whisper_model:Optional[str]=None) -> List[str]:
        """
        Process multiple audio parts.

        Args:
            parts: Sequence of audio parts to process
            username: Username for cost tracking

        Returns:
            List of transcription texts
        """
        chunks = []
        logger.info(f"Processing {len(parts)} audio parts")
        start_time = time.time()

        # this has to be done one by one, do NOT parallelize.
        for i, part in enumerate(parts):
            logger.info(f"Processing part {i+1}/{len(parts)}")
            # todo: make sure memory is freed after each part.
            chunks += await self.process_part(part, username=username, whisper_model=whisper_model)
            if isinstance(part, Path):
                if self.config.cleanup_downloads:
                    part.unlink(missing_ok=True)

        processing_time = time.time() - start_time
        logger.info(f"Processed {len(parts)} parts in {processing_time:.2f}s, got {len(chunks)} chunks")
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
        audio: Audio,
        username: Optional[str] = None
            , whisper_model:Optional[str]=None
    ) -> List[str]:
        """
        Process an audio part by splitting it into chunks and transcribing them.

        Args:
            audio: Audio data to process
            username: Username for cost tracking

        Returns:
            List of transcription texts
        """
        if isinstance(audio, (str, Path, BytesIO, BinaryIO)):
            logger.debug(f"Loading audio from {audio}")
            audio = AudioSegment.from_file(audio)
        assert isinstance(audio, AudioSegment)

        audio_duration = len(audio)
        logger.info(f"Processing audio part with duration: {audio_duration/1000:.2f} seconds")

        period = self._determine_optimal_period(
            audio_duration,
            target_chunk_count=self.config.target_chunk_count,
            max_chunk_size=self.config.maximum_chunk_duration,
            min_chunk_size=self.config.minimum_chunk_duration,
        )

        audio_chunks = split_audio(
            audio, period=period, buffer=self.config.overlap_duration
        )

        logger.info(f"Split audio into {len(audio_chunks)} chunks with period {period/1000:.2f} seconds")

        transcriptions = await self.parse_audio_chunks(
            audio_chunks,
            username=username,
            model_name=whisper_model or self.config.transcription_model
        )

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
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        username: Optional[str] = None,
    ) -> str:
        """
        Parse a single audio chunk using OpenAI Whisper API.

        Args:
            audio_path: Path to the audio file
            model_name: Name of the Whisper model to use (whisper-1)
            language: Language code (optional, auto-detected if None)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            username: Username for cost tracking

        Returns:
            Transcription text
        """
        if model_name is None:
            model_name = self.config.transcription_model

        logger.info(f"Transcribing {audio_file.name} using OpenAI Whisper API model {model_name}")
        start_time = time.time()

        async with self.whisper_semaphore:
            # Get audio duration for cost estimation
            audio_file.seek(0)
            audio_data = audio_file.read()
            audio_file.seek(0)

            # Estimate file size in MB for cost tracking
            file_size_mb = len(audio_data) / (1024 * 1024)

            # Call the OpenAI API
            options = {}
            if language:
                options["language"] = language

            response = await self.openai_client.audio.transcriptions.create(
                file=audio_file, model=model_name, response_format="text", **options
            )

            # Extract the transcription text
            transcription = response
            processing_time = time.time() - start_time

            # Log detailed information
            logger.info(
                f"Transcription complete for {audio_file.name}: {len(transcription)} characters in {processing_time:.2f}s"
            )

            # Estimate cost based on file size (approximate)
            # Whisper pricing: $0.006 per minute
            # Rough estimate: 1MB ≈ 1 minute of audio
            estimated_cost = file_size_mb * 0.006

            # Log cost information if username is provided
            if username:
                usage_info = {
                    "file_name": audio_file.name,
                    "file_size_mb": file_size_mb,
                    "processing_time": processing_time,
                    "characters": len(transcription),
                    "estimated_minutes": file_size_mb
                }

                await self.log_cost(
                    operation="transcription",
                    user_id=username,
                    model=model_name,
                    cost=estimated_cost,
                    usage=usage_info,
                    message_id=getattr(self, "_current_message_id", None)
                )

            return transcription

    async def parse_audio_chunks(
        self,
        audio_chunks: List[BytesIO],
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[str]:
        """
        Parse multiple audio chunks using OpenAI Whisper API.

        Args:
            audio_chunks: List of paths to audio files
            model_name: Name of the Whisper model to use (whisper-1)
            language: Language code (optional, auto-detected if None)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            max_concurrent: Maximum number of concurrent transcriptions
            username: Username for cost tracking

        Returns:
            List of transcription texts
        """
        logger.info(f"Transcribing {len(audio_chunks)} audio chunks using model {model_name}")
        start_time = time.time()

        # Create tasks for all chunks
        tasks = [
            self.parse_audio_chunk(
                chunk, 
                model_name=model_name, 
                language=language,
                username=username
            )
            for chunk in audio_chunks
        ]

        # Wait for all tasks to complete
        transcriptions = await asyncio.gather(*tasks)

        # Calculate and log total time taken
        total_time = time.time() - start_time
        avg_time_per_chunk = total_time / len(audio_chunks) if audio_chunks else 0

        logger.info(
            f"Transcription complete for all {len(audio_chunks)} chunks in {total_time:.2f}s "
            f"(avg: {avg_time_per_chunk:.2f}s per chunk)"
        )

        return transcriptions

    # endregion process_parts

    # region format_chunks_with_llm

    async def format_chunks_with_llm(
        self, chunks: List[str], username: str
    ) -> List[str]:
        """
        Format multiple text chunks using LLM.

        Args:
            chunks: List of text chunks to format
            username: Username for cost tracking

        Returns:
            List of formatted text chunks
        """
        logger.info(f"Formatting {len(chunks)} text chunks with LLM")
        start_time = time.time()

        # Create tasks for all chunks
        tasks = [self.format_chunk(chunk, username=username) for chunk in chunks]

        # Wait for all tasks to complete
        formatted_chunks = await asyncio.gather(*tasks)

        # Calculate and log total time taken
        total_time = time.time() - start_time
        avg_time_per_chunk = total_time / len(chunks) if chunks else 0

        logger.info(
            f"Formatting complete for all {len(chunks)} chunks in {total_time:.2f}s "
            f"(avg: {avg_time_per_chunk:.2f}s per chunk)"
        )

        return formatted_chunks

    async def format_chunk(self, chunk: str, username: str) -> str:
        """
        Format a single text chunk using LLM.

        Args:
            chunk: Text chunk to format
            username: Username for cost tracking

        Returns:
            Formatted text
        """
        logger.info(f"Formatting text chunk of length {len(chunk)} characters")
        start_time = time.time()

        async with self.format_semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

            formatted_text = await format_text_with_llm(
                text=chunk, 
                model=self.config.formatting_model, 
                username=username
            )

            processing_time = time.time() - start_time

            # Log detailed information
            logger.info(
                f"Formatting complete: {len(chunk)} → {len(formatted_text)} characters in {processing_time:.2f}s"
            )

            # Estimate cost based on token count (approximate)
            # Rough estimate: 1 token ≈ 4 characters
            input_tokens = len(chunk) / 4
            output_tokens = len(formatted_text) / 4

            # Cost estimation based on model
            model = self.config.formatting_model
            if "gpt-4" in model:
                # GPT-4 pricing: $0.01 per 1K input tokens, $0.03 per 1K output tokens
                estimated_cost = (input_tokens * 0.01 / 1000) + (output_tokens * 0.03 / 1000)
            else:
                # Default to GPT-3.5 pricing: $0.0015 per 1K input tokens, $0.002 per 1K output tokens
                estimated_cost = (input_tokens * 0.0015 / 1000) + (output_tokens * 0.002 / 1000)

            # Log cost information
            usage_info = {
                "input_length": len(chunk),
                "output_length": len(formatted_text),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "processing_time": processing_time
            }

            await self.log_cost(
                operation="formatting",
                user_id=username,
                model=self.config.formatting_model,
                cost=estimated_cost,
                usage=usage_info,
                message_id=getattr(self, "_current_message_id", None)
            )

            return formatted_text

    # endregion format_chunks_with_llm

    async def create_summary(self, transcript: str, username: str):
        """
        Create a summary of the transcript using the configured model.

        Args:
            transcript: Transcript text to summarize
            username: Username for cost tracking

        Returns:
            Summary text
        """
        logger.info(f"Creating summary of transcript with {len(transcript)} characters")
        start_time = time.time()

        async with self.summary_semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

            summary = await create_summary(
                transcription=transcript,
                username=username,
                model=self.config.summary_model,
                max_tokens=self.config.summary_max_tokens,
            )

            processing_time = time.time() - start_time

            # Log detailed information
            logger.info(
                f"Summary creation complete: {len(transcript)} → {len(summary)} characters in {processing_time:.2f}s"
            )

            # Estimate cost based on token count (approximate)
            # Rough estimate: 1 token ≈ 4 characters
            input_tokens = len(transcript) / 4
            output_tokens = len(summary) / 4

            # Cost estimation based on model
            model = self.config.summary_model
            if "claude-4" in model:
                # Claude-4 pricing: $0.015 per 1K input tokens, $0.075 per 1K output tokens
                estimated_cost = (input_tokens * 0.015 / 1000) + (output_tokens * 0.075 / 1000)
            elif "claude-3" in model:
                # Claude-3 pricing: $0.008 per 1K input tokens, $0.024 per 1K output tokens
                estimated_cost = (input_tokens * 0.008 / 1000) + (output_tokens * 0.024 / 1000)
            else:
                # Default to GPT-4 pricing: $0.01 per 1K input tokens, $0.03 per 1K output tokens
                estimated_cost = (input_tokens * 0.01 / 1000) + (output_tokens * 0.03 / 1000)

            # Log cost information
            usage_info = {
                "input_length": len(transcript),
                "output_length": len(summary),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "processing_time": processing_time
            }

            # Log event for summary creation
            await self.log_event("summary_creation", username, {
                "input_length": len(transcript),
                "output_length": len(summary),
                "processing_time": processing_time
            })

            await self.log_cost(
                operation="summary",
                user_id=username,
                model=self.config.summary_model,
                cost=estimated_cost,
                usage=usage_info,
                message_id=getattr(self, "_current_message_id", None)
            )

            return summary

    async def chat_about_transcript(
        self, full_prompt: str, username: str
    ) -> str:
        """
        Chat about the transcript using the configured chat model.

        Args:
            full_prompt: Full prompt text including the transcript and user query
            username: Username for cost tracking

        Returns:
            Response text from the chat model
        """
        logger.info(f"Processing chat request with {len(full_prompt)} characters")
        start_time = time.time()

        # Log chat event
        await self.log_event("chat_request", username, {
            "prompt_length": len(full_prompt)
        })

        async with self.chat_semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

            response = await aquery_llm_text(
                prompt=full_prompt,
                user=username,
                model=self.config.chat_model,
                max_tokens=self.config.chat_max_tokens,
            )

            processing_time = time.time() - start_time

            # Log detailed information
            logger.info(
                f"Chat response complete: {len(full_prompt)} → {len(response)} characters in {processing_time:.2f}s"
            )

            # Estimate cost based on token count (approximate)
            # Rough estimate: 1 token ≈ 4 characters
            input_tokens = len(full_prompt) / 4
            output_tokens = len(response) / 4

            # Cost estimation based on model
            model = self.config.chat_model
            if "claude-4" in model:
                # Claude-4 pricing: $0.015 per 1K input tokens, $0.075 per 1K output tokens
                estimated_cost = (input_tokens * 0.015 / 1000) + (output_tokens * 0.075 / 1000)
            elif "claude-3" in model:
                # Claude-3 pricing: $0.008 per 1K input tokens, $0.024 per 1K output tokens
                estimated_cost = (input_tokens * 0.008 / 1000) + (output_tokens * 0.024 / 1000)
            else:
                # Default to GPT-4 pricing: $0.01 per 1K input tokens, $0.03 per 1K output tokens
                estimated_cost = (input_tokens * 0.01 / 1000) + (output_tokens * 0.03 / 1000)

            # Log cost information
            usage_info = {
                "input_length": len(full_prompt),
                "output_length": len(response),
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "processing_time": processing_time
            }

            # Log completion event
            await self.log_event("chat_completion", username, {
                "input_length": len(full_prompt),
                "output_length": len(response),
                "processing_time": processing_time
            })

            await self.log_cost(
                operation="chat",
                user_id=username,
                model=self.config.chat_model,
                cost=estimated_cost,
                usage=usage_info,
                message_id=getattr(self, "_current_message_id", None)
            )

            return response
