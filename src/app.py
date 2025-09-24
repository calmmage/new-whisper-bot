import asyncio
import datetime
import time
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

import openai
from aiogram.types import Message as AiogramMessage
from botspot.components.data.mongo_database import get_database
from botspot.components.new.llm_provider import aquery_llm_text
from botspot.core.errors import BotspotError
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

from src.utils.audio_utils import Audio, split_audio
from src.utils.convert_to_mp3_ffmpeg import convert_to_mp3_ffmpeg
from src.utils.cost_tracking import (
    create_usage_info,
    estimate_cost_from_text,
    estimate_whisper_cost,
)
from src.utils.create_summary import create_summary
from src.utils.cut_audio_ffmpeg import cut_audio_ffmpeg
from src.utils.download_attachment import download_file
from src.utils.format_text_with_llm import format_text_with_llm
from src.utils.text_utils import merge_all_chunks


class AppConfig(BaseSettings):
    telegram_api_id: int
    telegram_api_hash: SecretStr
    telegram_bot_token: SecretStr

    downloads_dir: Path = Path("downloads").absolute()
    cleanup_downloads: bool = True
    # todo: allow user to configure
    cleanup_messages: bool = False  # delete messages after processing
    use_original_file_name: bool = False

    # Disable Pydub speedup due to poor quality (chipmunk effect)
    disable_pydub_speedup: bool = True

    # cutting parameters
    target_part_size: int = (
        50 * 1024 * 1024
    )  # 25 * 1024 * 1024  # desired max size in bytes
    target_chunk_count: int = 10  # 20
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
    summary_generation_threshold: int = 4000
    # todo: allow user to configure
    chat_model: str = (
        # "claude-4-sonnet"  # Default chat model for discussing transcripts and summaries
        "gpt-4.1"  # I got free quote on this one
        # "grok-3" # I had free quota before but not anymore..
    )
    chat_max_tokens: int = 2048
    # todo: allow user to configure
    formatting_model: str = "gpt-4.1-nano"
    formatting_disable_threshold: int = (
        10000  # Skip formatting if total chunks length exceeds this
    )

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

        # Per-user message tracking for async safety
        self._user_message_ids: Dict[str, Optional[int]] = {}

        # Accumulated costs for batch processing
        self._accumulated_costs: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def db(self):
        if self._db is None:
            self._db = get_database()
        return self._db

    def _get_user_message_id(self, username: str) -> Optional[int]:
        """Get the current message ID for a user."""
        return self._user_message_ids.get(username)

    async def track_costs(
        self,
        operation: str,
        username: Optional[str],
        model: str,
        input_data: Union[str, bytes, AudioSegment],
        output_data: str,
        processing_time: float,
        message_id: Optional[int] = None,
        file_name: Optional[str] = None,
        audio_duration_seconds: Optional[float] = None,
        write_to_db: bool = False,
    ) -> None:
        """
        Track costs for an operation.

        Args:
            operation: Type of operation (transcription, formatting, summary, chat)
            username: Username for cost tracking
            model: Model used
            input_data: Input data (text, audio bytes, or AudioSegment)
            output_data: Output text
            processing_time: Time taken to process
            file_name: Optional file name
            audio_duration_seconds: Optional audio duration in seconds
            write_to_db: Whether to write to the database immediately (default: False)
        """
        if not username:
            logger.warning("Username is None, skipping cost tracking.")
            return

        if message_id is None:
            message_id = self._get_user_message_id(username)

        # Calculate input and output lengths
        if isinstance(input_data, AudioSegment):
            input_length = len(input_data)  # Duration in ms
            file_size_mb = None
            # For Whisper, "tokens" = minutes
            estimated_minutes = (
                audio_duration_seconds / 60
                if audio_duration_seconds is not None
                else input_length / (60 * 1000)
            )
            estimated_input_tokens = estimated_minutes
            estimated_output_tokens = 0
        elif isinstance(input_data, bytes):
            input_length = len(input_data)
            file_size_mb = input_length / (1024 * 1024)
            # For Whisper, "tokens" = minutes
            estimated_minutes = (
                audio_duration_seconds / 60
                if audio_duration_seconds is not None
                else file_size_mb
            )
            estimated_input_tokens = estimated_minutes
            estimated_output_tokens = 0
        else:
            input_length = len(input_data)
            file_size_mb = None
            # Rough estimate: 1 token ≈ 4 characters
            estimated_input_tokens = input_length / 4
            estimated_output_tokens = len(output_data) / 4

        output_length = len(output_data)

        # Estimate cost
        if operation == "transcription":
            assert (
                audio_duration_seconds is not None
            ), "audio_duration_seconds is required for transcription cost estimation"
            estimated_cost = estimate_whisper_cost(
                file_size_mb or (input_length / (1024 * 1024)),
                model,
                duration_seconds=audio_duration_seconds,
            )
        else:
            estimated_cost = estimate_cost_from_text(
                input_data if isinstance(input_data, str) else "", output_data, model
            )

        # Create usage info
        usage_info = create_usage_info(
            input_length=input_length,
            output_length=output_length,
            processing_time=processing_time,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            file_name=file_name,
            file_size_mb=file_size_mb,
            audio_duration_seconds=audio_duration_seconds,
        )

        # Create cost data
        cost_data = {
            "operation": operation,
            "user_id": username,
            "model": model,
            "cost": estimated_cost,
            "usage": usage_info,
            "message_id": self._get_user_message_id(username),
            "timestamp": datetime.datetime.now(),
        }

        # Add to accumulated costs
        if username not in self._accumulated_costs:
            self._accumulated_costs[username] = []
        self._accumulated_costs[username].append(cost_data)

        # Log cost to database if requested
        if write_to_db:
            await self.log_cost(
                operation=operation,
                user_id=username,
                model=model,
                cost=estimated_cost,
                usage=usage_info,
                message_id=message_id,
            )

    async def write_accumulated_costs(
        self, username: str, operation_type: Optional[str] = None
    ) -> None:
        """
        Write accumulated costs to the database and clear the accumulator.

        Args:
            username: Username for cost tracking
            operation_type: Optional operation type to filter costs (e.g., 'transcription', 'formatting')
        """
        if (
            username not in self._accumulated_costs
            or not self._accumulated_costs[username]
        ):
            return

        costs_to_write = self._accumulated_costs[username]

        # Filter by operation type if specified
        if operation_type:
            costs_to_write = [
                cost for cost in costs_to_write if cost["operation"] == operation_type
            ]

        if not costs_to_write:
            return

        # Calculate total cost and usage for each operation type
        operation_totals = {}
        for cost_data in costs_to_write:
            operation = cost_data["operation"]
            if operation not in operation_totals:
                operation_totals[operation] = {
                    "cost": 0.0,
                    "input_length": 0,
                    "output_length": 0,
                    "processing_time": 0.0,
                    "count": 0,
                    "model": cost_data["model"],
                    "message_id": cost_data["message_id"],
                }

            operation_totals[operation]["cost"] += cost_data["cost"]
            operation_totals[operation]["input_length"] += cost_data["usage"].get(
                "input_length", 0
            )
            operation_totals[operation]["output_length"] += cost_data["usage"].get(
                "output_length", 0
            )
            operation_totals[operation]["processing_time"] += cost_data["usage"].get(
                "processing_time", 0.0
            )
            operation_totals[operation]["count"] += 1

        # Write aggregated costs to database
        for operation, totals in operation_totals.items():
            usage_info = {
                "input_length": totals["input_length"],
                "output_length": totals["output_length"],
                "processing_time": totals["processing_time"],
                "count": totals["count"],
            }

            await self.log_cost(
                operation=operation,
                user_id=username,
                model=totals["model"],
                cost=totals["cost"],
                usage=usage_info,
                message_id=totals["message_id"],
            )

        # Clear the accumulated costs for this user and operation
        if operation_type:
            self._accumulated_costs[username] = [
                cost
                for cost in self._accumulated_costs[username]
                if cost["operation"] != operation_type
            ]
        else:
            self._accumulated_costs[username] = []

        logger.info(f"Wrote accumulated costs for user {username} to database")

    # todo: rework all our tracking methods to capture full input and output of each operation
    #  at the per-operation level, not per-chunk.
    async def log_event(
        self, event_type: str, user_id: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
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
            "timestamp": datetime.datetime.now(),
            **data,
        }

        logger.info(f"Logging event: {event_type} for user {user_id}")
        await self.db.events.insert_one(event)

    async def log_cost(
        self,
        operation: str,
        user_id: str,
        model: str,
        cost: Optional[float] = None,
        usage: Optional[Dict[str, Any]] = None,
        message_id: Optional[int] = None,
    ) -> None:
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
            "timestamp": datetime.datetime.now(),
            "usage": usage,
        }

        if cost is not None:
            cost_data["cost"] = cost

        if message_id is not None:
            cost_data["message_id"] = str(message_id)

        logger.info(f"Logging cost for {operation} using {model} for user {user_id}")
        await self.db.costs.insert_one(cost_data)

    async def get_total_cost(
        self, user_id: str, message_id: Optional[int] = None
    ) -> Dict[str, Any]:
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
            query["message_id"] = str(message_id)

        costs = await self.db.costs.find(query).to_list(length=1000)

        total_cost = 0.0
        operation_costs = {}
        model_costs = {}

        for cost_entry in costs:
            cost_raw = cost_entry.get("cost", 0.0)
            # Handle both string and float costs for backward compatibility
            cost = float(cost_raw) if cost_raw is not None else 0.0
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
            "cost_count": len(costs),
        }

    # Main method
    async def process_message(
        self,
        message: AiogramMessage,
        state,  # FSMContext
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
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
            "file_type": self._get_file_type(message),
        }
        logger.info(f"User {username} submitted a file for processing: {file_info}")
        await self.log_event("file_submission", username, file_info)

        media_file = await self.download_attachment(
            message, status_callback=status_callback
        )

        parts = await self.prepare_parts(media_file, message, state, status_callback=status_callback)

        # Store message_id per user for async safety
        self._user_message_ids[username] = message_id

        chunks = await self.process_parts(
            parts,
            username=username,
            whisper_model=whisper_model,
            language=language,
            status_callback=status_callback,
        )

        chunks = await self.format_chunks_with_llm(
            chunks, username=username, status_callback=status_callback
        )

        result = merge_all_chunks(chunks)

        # Log file processing completion
        processing_time = time.time() - start_time
        completion_info = {
            "message_id": message_id,
            "processing_time_seconds": processing_time,
            "result_length": len(result),
            "chunks_count": len(chunks),
        }
        logger.info(
            f"Completed processing file for user {username} in {processing_time:.2f} seconds"
        )
        await self.log_event("file_processing_complete", username, completion_info)

        # Clear the user's message ID
        self._user_message_ids.pop(username, None)

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
        self,
        message: AiogramMessage,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Union[BinaryIO, Path]:
        if status_callback is not None:
            await status_callback("Downloading media file...")
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
        self,
        media_file: Union[BinaryIO, Path],
        message: AiogramMessage,
        state,  # FSMContext
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Sequence[Audio]:
        if isinstance(media_file, Path):
            # process file on disk - FFmpeg speedup available, ask user
            from src.router import ask_user_speedup
            speedup = await ask_user_speedup(message, self, state)
            parts = await self.process_file_on_disk(media_file, speedup=speedup, status_callback=status_callback)
            if self.config.cleanup_downloads:
                if media_file != parts[0]:
                    media_file.unlink(missing_ok=True)
            return parts
        else:
            # process file in memory - no speedup available (voice notes)
            assert isinstance(media_file, (BinaryIO, BytesIO))
            if not self.config.disable_pydub_speedup:
                speedup = await ask_user_speedup(message, self, state)
                if status_callback is not None:
                    await status_callback(f"Applying {speedup}x speedup to audio...")
                # Apply speedup using pydub
                media_file = await self._apply_speedup_pydub(media_file, speedup)
            if status_callback is not None:
                await status_callback(
                    "Just one part to process\n<b>Estimated processing time: Should be up to 1 minute</b>"
                )
            return [media_file]

    async def process_file_on_disk(self, media_file: Path, speedup: Optional[float] = None, status_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> Sequence[Audio]:
        # Skip conversion only if it's already MP3 AND there's no speedup
        skip_conversion = media_file.suffix == ".mp3" and speedup is None
        
        if not skip_conversion:
            if status_callback is not None:
                status_msg = "Preparing audio - converting to mp3"
                if speedup is not None:
                    status_msg += f" with {speedup}x speedup"
                status_msg += ".."
                await status_callback(status_msg)
            
            mp3_file = await convert_to_mp3_ffmpeg(media_file, speedup=speedup)
            # delete original file
            if self.config.cleanup_downloads:
                media_file.unlink(missing_ok=True)
            media_file = mp3_file

        parts = await self.cut_audio_with_ffmpeg(media_file, status_callback=status_callback)

        return parts

    async def cut_audio_with_ffmpeg(self, media_file, status_callback: Optional[Callable[[str], Awaitable[None]]] = None) -> Sequence[Audio]:
        file_size = self._get_file_size(media_file)
        num_parts = int(file_size / self.config.target_part_size) + 1

        logger.info(
            f"Cutting file sized {file_size / (1024 * 1024):.2f} MB into {num_parts} parts - target size {self.config.target_part_size / (1024 * 1024):.2f} MB"
        )
        if num_parts > 1:
            if status_callback is not None:
                await status_callback("Preparing audio - cutting into parts..\n<b>Estimated processing time: Several minutes</b>")
            parts = await cut_audio_ffmpeg(media_file, num_parts=num_parts)
            if self.config.cleanup_downloads:
                media_file.unlink(missing_ok=True)
            return parts
        else:
            if status_callback is not None:
                await status_callback(
                    "Just one part to process\n<b>Estimated processing time: Should be about 1-2 minutes</b>"
                )
            return [media_file]

    @staticmethod
    def _get_file_size(media_file):
        return media_file.stat().st_size

    async def _apply_speedup_pydub(self, media_file: Union[BinaryIO, BytesIO], speedup: float) -> BytesIO:
        """Apply speedup to audio using pydub."""
        import asyncio
        
        def _speedup_audio():
            # Load audio from binary data
            audio = AudioSegment.from_file(media_file)
            
            # Apply speedup using frame rate manipulation to preserve pitch quality
            # This avoids the chipmunk effect by manipulating frame rate then normalizing
            sound_with_altered_frame_rate = audio._spawn(
                audio.raw_data,
                overrides={"frame_rate": int(audio.frame_rate * speedup)}
            )
            # Convert back to original frame rate to avoid pitch shift
            faster_audio = sound_with_altered_frame_rate.set_frame_rate(audio.frame_rate)
            
            # Convert back to BytesIO
            output_buffer = BytesIO()
            faster_audio.export(output_buffer, format="mp3")
            output_buffer.seek(0)
            return output_buffer
        
        # Run in thread to avoid blocking
        return await asyncio.get_event_loop().run_in_executor(None, _speedup_audio)

    # endregion prepare_parts

    # region process_parts
    async def process_parts(
        self,
        parts: Sequence[Audio],
        username: Optional[str] = None,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> List[str]:
        """
        Process multiple audio parts.

        Args:
            parts: Sequence of audio parts to process
            username: Username for cost tracking
            whisper_model: Optional model to use for transcription

        Returns:
            List of transcription texts
        """
        chunks = []
        logger.info(f"Processing {len(parts)} audio parts")
        start_time = time.time()

        if status_callback is not None:
            await status_callback(f"Parsing audio - {len(parts)} parts...")

        # this has to be done one by one, do NOT parallelize.
        for i, part in enumerate(parts):
            logger.info(f"Processing part {i + 1}/{len(parts)}")
            # todo: make sure memory is freed after each part.
            chunks += await self.process_part(
                part, username=username, whisper_model=whisper_model, language=language
            )
            if status_callback is not None:
                await status_callback(f"Part {i + 1}/{len(parts)} done")
            if isinstance(part, Path):
                if self.config.cleanup_downloads:
                    part.unlink(missing_ok=True)

        processing_time = time.time() - start_time
        logger.info(
            f"Processed {len(parts)} parts in {processing_time:.2f}s, got {len(chunks)} chunks"
        )

        # Write accumulated transcription costs to the database
        if username:
            await self.write_accumulated_costs(username, operation_type="transcription")

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
        username: Optional[str] = None,
        whisper_model: Optional[str] = None,
        language: Optional[str] = None,
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
        logger.info(
            f"Processing audio part with duration: {audio_duration / 1000:.2f} seconds"
        )

        period = self._determine_optimal_period(
            audio_duration,
            target_chunk_count=self.config.target_chunk_count,
            max_chunk_size=self.config.maximum_chunk_duration,
            min_chunk_size=self.config.minimum_chunk_duration,
        )

        audio_chunks = split_audio(
            audio,
            period=period,
            buffer=self.config.overlap_duration,
            return_as_files=False,
        )
        assert all(
            isinstance(audio_chunk, AudioSegment) for audio_chunk in audio_chunks
        )

        logger.info(
            f"Split audio into {len(audio_chunks)} chunks with period {period / 1000:.2f} seconds"
        )

        transcriptions = await self.parse_audio_chunks(
            audio_chunks,
            username=username,
            model_name=whisper_model or self.config.transcription_model,
            language=language,
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
        audio_chunk: AudioSegment,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        username: Optional[str] = None,
        chunk_index: int = 0,
    ) -> str:
        """
        Parse a single audio chunk using OpenAI Whisper API.

        Args:
            audio_chunk: Audio segment to transcribe
            model_name: Name of the Whisper model to use (whisper-1)
            language: Language code (optional, auto-detected if None)
            username: Username for cost tracking
            chunk_index: Index of the chunk for naming

        Returns:
            Transcription text
        """
        if model_name is None:
            model_name = self.config.transcription_model

        chunk_name = f"chunk_{chunk_index}.mp3"
        logger.info(
            f"Transcribing {chunk_name} using OpenAI Whisper API model {model_name}"
        )
        start_time = time.time()

        async with self.whisper_semaphore:
            # Get audio duration directly from AudioSegment
            # Duration in milliseconds
            audio_duration_ms = len(audio_chunk)
            # Convert to seconds
            audio_duration_sec = audio_duration_ms / 1000
            logger.info(f"Audio duration: {audio_duration_sec:.2f} seconds")

            # Convert AudioSegment to BytesIO for API call
            audio_file = BytesIO()
            audio_chunk.export(audio_file, format="mp3")
            audio_file.name = chunk_name
            audio_file.seek(0)

            # Get audio data for file size calculation
            audio_file.seek(0)
            audio_data = audio_file.read()
            audio_file.seek(0)

            # Estimate file size in MB for cost tracking
            # file_size_mb = len(audio_data) / (1024 * 1024)

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

            # Track costs (accumulate without writing to DB)
            await self.track_costs(
                operation="transcription",
                username=username,
                model=model_name,
                input_data=audio_data,
                output_data=transcription,
                processing_time=processing_time,
                file_name=audio_file.name,
                audio_duration_seconds=audio_duration_sec,
                write_to_db=False,
            )

            return transcription

    async def parse_audio_chunks(
        self,
        audio_chunks: List[AudioSegment],
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        username: Optional[str] = None,
    ) -> List[str]:
        """
        Parse multiple audio chunks using OpenAI Whisper API.

        Args:
            audio_chunks: List of audio segments
            model_name: Name of the Whisper model to use (whisper-1)
            language: Language code (optional, auto-detected if None)
            username: Username for cost tracking

        Returns:
            List of transcription texts
        """
        logger.info(
            f"Transcribing {len(audio_chunks)} audio chunks using model {model_name}"
        )
        start_time = time.time()

        # Create tasks for all chunks
        tasks = [
            self.parse_audio_chunk(
                chunk,
                model_name=model_name,
                language=language,
                username=username,
                chunk_index=i,
            )
            for i, chunk in enumerate(audio_chunks)
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
        self,
        chunks: List[str],
        username: str,
        status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> List[str]:
        """
        Format multiple text chunks using LLM.

        Args:
            chunks: List of text chunks to format
            username: Username for cost tracking

        Returns:
            List of formatted text chunks
        """
        # Check total chunks length against threshold
        total_length = sum(len(chunk) for chunk in chunks)

        if total_length > self.config.formatting_disable_threshold:
            logger.info(
                f"Skipping formatting: total chunks length ({total_length} chars) "
                f"exceeds threshold ({self.config.formatting_disable_threshold} chars)"
            )
            return chunks

        if status_callback is not None:
            await status_callback("Formatting punctuation and capitalization...")

        logger.info(
            f"Formatting {len(chunks)} text chunks with LLM (total: {total_length} chars)"
        )
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

        # Write accumulated formatting costs to the database
        await self.write_accumulated_costs(username, operation_type="formatting")

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
                text=chunk, model=self.config.formatting_model, username=username
            )

            processing_time = time.time() - start_time

            # Log detailed information
            logger.info(
                f"Formatting complete: {len(chunk)} → {len(formatted_text)} characters in {processing_time:.2f}s"
            )

            # Track costs (accumulate without writing to DB)
            await self.track_costs(
                operation="formatting",
                username=username,
                model=self.config.formatting_model,
                input_data=chunk,
                output_data=formatted_text,
                processing_time=processing_time,
                write_to_db=False,
            )

            return formatted_text

    # endregion format_chunks_with_llm

    async def create_summary(
        self, transcript: str, username: str, message_id: Optional[int] = None
    ) -> str:
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

            # Set message_id for this user
            self._user_message_ids[username] = message_id

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

            # Log event for summary creation
            await self.log_event(
                "summary_creation",
                username,
                {
                    "input_length": len(transcript),
                    "output_length": len(summary),
                    "processing_time": processing_time,
                },
            )

            # Track costs (accumulate without writing to DB)
            await self.track_costs(
                operation="summary",
                username=username,
                model=self.config.summary_model,
                input_data=transcript,
                output_data=summary,
                processing_time=processing_time,
                write_to_db=True,
            )

            # Write accumulated summary costs to the database
            # todo: check that the operation above with write_to_db=True is equivalent to this
            # await self.write_accumulated_costs(username, operation_type="summary")

            return summary

    async def chat_about_transcript(
        self, full_prompt: str, username: str, model: str = None, message_id: Optional[int] = None
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

        if model is None:
            model = self.config.chat_model

        # Log chat event
        await self.log_event(
            "chat_request", username, {"prompt_length": len(full_prompt)}
        )

        async with self.chat_semaphore:
            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

            response = await aquery_llm_text(
                prompt=full_prompt,
                user=username,
                model=model,
                max_tokens=self.config.chat_max_tokens,
            )

            processing_time = time.time() - start_time

            # Log detailed information
            logger.info(
                f"Chat response complete: {len(full_prompt)} → {len(response)} characters in {processing_time:.2f}s"
            )

            # Log completion event
            await self.log_event(
                "chat_completion",
                username,
                {
                    "input_length": len(full_prompt),
                    "output_length": len(response),
                    "processing_time": processing_time,
                },
            )

            # Track costs (accumulate without writing to DB)
            await self.track_costs(
                operation="chat",
                username=username,
                model=self.config.chat_model,
                message_id=message_id,
                input_data=full_prompt,
                output_data=response,
                processing_time=processing_time,
                write_to_db=True,
            )

            return response

    async def get_user_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics for all users.

        Returns:
            Dictionary with user statistics including request counts, total minutes, and costs
        """
        # Get all cost entries from the database
        costs = await self.db.costs.find({}).to_list(length=10000)
        events = await self.db.events.find({"event_type": "file_submission"}).to_list(length=10000)
        
        # Initialize user stats dictionary
        user_stats = {}
        
        # Process cost data
        for cost_entry in costs:
            user_id = cost_entry.get("user_id")
            if not user_id:
                continue
                
            if user_id not in user_stats:
                user_stats[user_id] = {
                    "total_cost": 0.0,
                    "total_requests": 0,
                    "total_minutes": 0.0,
                    "operations": {},
                    "models": {},
                    "first_activity": None,
                    "last_activity": None,
                }
            
            # Add cost
            cost = float(cost_entry.get("cost", 0.0)) if cost_entry.get("cost") is not None else 0.0
            user_stats[user_id]["total_cost"] += cost
            
            # Track operation types
            operation = cost_entry.get("operation", "unknown")
            if operation not in user_stats[user_id]["operations"]:
                user_stats[user_id]["operations"][operation] = {"cost": 0.0, "count": 0}
            user_stats[user_id]["operations"][operation]["cost"] += cost
            user_stats[user_id]["operations"][operation]["count"] += 1
            
            # Track models used
            model = cost_entry.get("model", "unknown")
            if model not in user_stats[user_id]["models"]:
                user_stats[user_id]["models"][model] = {"cost": 0.0, "count": 0}
            user_stats[user_id]["models"][model]["cost"] += cost
            user_stats[user_id]["models"][model]["count"] += 1
            
            # Track timestamps
            timestamp = cost_entry.get("timestamp")
            if timestamp:
                if user_stats[user_id]["first_activity"] is None or timestamp < user_stats[user_id]["first_activity"]:
                    user_stats[user_id]["first_activity"] = timestamp
                if user_stats[user_id]["last_activity"] is None or timestamp > user_stats[user_id]["last_activity"]:
                    user_stats[user_id]["last_activity"] = timestamp
            
            # Extract minutes from usage data
            usage = cost_entry.get("usage", {})
            if isinstance(usage, dict):
                # For transcription operations, get minutes from estimated_minutes or audio_duration_seconds
                if operation == "transcription":
                    estimated_minutes = usage.get("estimated_minutes", 0)
                    if estimated_minutes:
                        user_stats[user_id]["total_minutes"] += float(estimated_minutes)
                    elif usage.get("audio_duration_seconds"):
                        minutes = float(usage.get("audio_duration_seconds", 0)) / 60
                        user_stats[user_id]["total_minutes"] += minutes
        
        # Process event data to get request counts
        for event in events:
            user_id = event.get("user_id")
            if user_id and user_id in user_stats:
                user_stats[user_id]["total_requests"] += 1
        
        # Calculate summary statistics
        total_users = len(user_stats)
        total_cost_all_users = sum(stats["total_cost"] for stats in user_stats.values())
        total_requests_all_users = sum(stats["total_requests"] for stats in user_stats.values())
        total_minutes_all_users = sum(stats["total_minutes"] for stats in user_stats.values())
        
        return {
            "user_stats": user_stats,
            "summary": {
                "total_users": total_users,
                "total_cost": total_cost_all_users,
                "total_requests": total_requests_all_users,
                "total_minutes": total_minutes_all_users,
            }
        }
