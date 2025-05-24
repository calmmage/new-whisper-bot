import threading
import time
from pathlib import Path

import psutil
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from botspot import commands_menu, reply_safe
from botspot.utils import send_safe
from botspot.utils.unsorted import get_message_attachments
from loguru import logger

from app import App

router = Router()

SAMPLE_DATA_DIR = Path("./sample_data")
SAMPLE_DATA_DIR.mkdir(exist_ok=True)


@commands_menu.botspot_command("start", "Start the bot")
@router.message(CommandStart())
async def start_handler(message: Message, app: App):
    await send_safe(message.chat.id, f"Hello! Welcome to {app.name}!")


@commands_menu.botspot_command("help", "Show this help message")
@router.message(Command("help"))
async def help_handler(message: Message, app: App):
    """Basic help command handler"""
    await send_safe(message.chat.id, f"This is {app.name}. Use /start to begin.")


def get_file_info(attachment):
    """
    Extract file name and mime type from aiogram attachment (Audio, Voice, Video, Document).
    If file_name is missing (e.g. for voice), generate a default name with the correct extension.
    """
    file_name = getattr(attachment, "file_name", None)
    mime_type = getattr(attachment, "mime_type", None)
    logger.info(
        f"Extracting file info: file_name={file_name}, mime_type={mime_type}, type={type(attachment)}"
    )
    # Handle voice messages (no file_name)
    if file_name is None:
        if hasattr(attachment, "mime_type") and attachment.mime_type == "audio/ogg":
            file_name = "voice_message.ogg"
        elif hasattr(attachment, "mime_type") and attachment.mime_type:
            ext = attachment.mime_type.split("/")[-1]
            file_name = f"file.{ext}"
        else:
            file_name = "file.bin"
    logger.info(f"Final file_name: {file_name}, mime_type: {mime_type}")
    return file_name, mime_type


def monitor_process_memory(process, interval=5.0):
    """Monitor memory usage of a process and its children"""
    memory_stats = []

    def monitor():
        try:
            while process.poll() is None:  # While process is running
                try:
                    # Get memory info for main process
                    memory_info = process.memory_info()

                    # Get memory info for all children
                    children_memory = 0
                    try:
                        psutil_process = psutil.Process(process.pid)
                        for child in psutil_process.children(recursive=True):
                            try:
                                children_memory += child.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                    total_memory_mb = (memory_info.rss + children_memory) / 1024 / 1024
                    system_memory = psutil.virtual_memory()

                    stats = {
                        "timestamp": time.time(),
                        "process_memory_mb": memory_info.rss / 1024 / 1024,
                        "children_memory_mb": children_memory / 1024 / 1024,
                        "total_memory_mb": total_memory_mb,
                        "system_available_mb": system_memory.available / 1024 / 1024,
                        "system_used_percent": system_memory.percent,
                    }
                    memory_stats.append(stats)

                    logger.info(
                        f"Memory: Process={stats['process_memory_mb']:.1f}MB, "
                        f"Children={stats['children_memory_mb']:.1f}MB, "
                        f"Total={stats['total_memory_mb']:.1f}MB, "
                        f"System={stats['system_used_percent']:.1f}%"
                    )

                except Exception as e:
                    logger.warning(f"Error monitoring memory: {e}")

                time.sleep(interval)
        except Exception as e:
            logger.error(f"Memory monitoring thread error: {e}")

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return memory_stats


def convert_video_to_audio(video_path: Path) -> Path:
    """Convert video file to mp3 using ffmpeg with memory monitoring."""
    import subprocess

    audio_path = video_path.with_suffix(".mp3")

    # Log initial system memory state
    system_memory = psutil.virtual_memory()
    logger.info(
        f"Starting conversion. System memory: {system_memory.used / 1024 / 1024:.1f}MB used "
        f"({system_memory.percent:.1f}%), {system_memory.available / 1024 / 1024:.1f}MB available"
    )

    # Log file size
    file_size_mb = video_path.stat().st_size / 1024 / 1024
    logger.info(f"Input file size: {file_size_mb:.1f}MB")

    logger.info(f"Converting {video_path} to {audio_path} using ffmpeg...")

    # Start ffmpeg process
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ab",
            "128k",  # Set bitrate for more predictable output size
            str(audio_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Start memory monitoring
    memory_stats = monitor_process_memory(process, interval=2.0)

    # Wait for completion
    start_time = time.time()
    stdout, stderr = process.communicate()
    end_time = time.time()

    processing_time = end_time - start_time

    if process.returncode != 0:
        logger.error(f"ffmpeg failed: {stderr.decode()}")
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

    # Log final statistics
    output_size_mb = audio_path.stat().st_size / 1024 / 1024
    final_system_memory = psutil.virtual_memory()

    if memory_stats:
        max_memory = max(stat["total_memory_mb"] for stat in memory_stats)
        avg_memory = sum(stat["total_memory_mb"] for stat in memory_stats) / len(
            memory_stats
        )

        logger.info(f"Conversion completed in {processing_time:.1f}s")
        logger.info(
            f"Memory usage - Peak: {max_memory:.1f}MB, Average: {avg_memory:.1f}MB"
        )
        logger.info(
            f"File sizes - Input: {file_size_mb:.1f}MB, Output: {output_size_mb:.1f}MB"
        )
        logger.info(f"Compression ratio: {output_size_mb / file_size_mb:.2f}")

        # Estimate for 4-hour files
        if processing_time > 0:
            time_ratio = processing_time / (file_size_mb / 1024)  # seconds per GB
            logger.info(
                f"Performance: {time_ratio:.1f}s processing time per GB of input"
            )

            # Estimate for 4-hour video (assuming ~1GB per hour for decent quality)
            estimated_4h_size_gb = 4.0
            estimated_4h_processing_time = time_ratio * estimated_4h_size_gb
            estimated_4h_memory_mb = max_memory * (
                estimated_4h_size_gb / (file_size_mb / 1024)
            )

            logger.info(
                f"4-hour file estimates: ~{estimated_4h_processing_time / 60:.1f} minutes processing, "
                f"~{estimated_4h_memory_mb:.0f}MB peak memory"
            )

    logger.info(
        f"System memory after conversion: {final_system_memory.used / 1024 / 1024:.1f}MB used "
        f"({final_system_memory.percent:.1f}%)"
    )
    logger.info(f"Audio saved to {audio_path}")
    return audio_path


@router.message(F.audio | F.voice | F.video | F.document | F.video_note)
async def media_downloader_handler(message: Message, app: App, state: FSMContext):
    logger.info(
        f"media_downloader_handler: received message {message.message_id} from chat {message.chat.id}"
    )

    # Use the app's run method to process the media
    transcription = await app.run(message.message_id, message.from_user.id)
    # logger.info(f"Transcription completed successfully: {transcription[:100]}...")

    await reply_safe(message, transcription)
