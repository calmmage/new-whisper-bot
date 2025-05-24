import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import psutil
from loguru import logger


async def convert_video_to_audio(
    video_path: Path, 
    output_path: Optional[Path] = None, 
    format: str = "mp3",
    use_memory_profiler: bool = False
) -> Path:
    """
    Convert video file to audio using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file (optional)
        format: Audio format (default: mp3)
        use_memory_profiler: Whether to use memory profiler implementation

    Returns:
        Path to the converted audio file
    """
    # If the file is already an audio file, return it as is
    if video_path.suffix.lower() in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]:
        logger.info(f"File {video_path} is already an audio file, skipping conversion")
        return video_path

    # If output_path is not provided, use the same directory and name as the video file
    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose the appropriate implementation based on the flag
    if use_memory_profiler:
        return await convert_video_to_audio_with_profiler(video_path, output_path, format)
    else:
        return await convert_video_to_audio_standard(video_path, output_path, format)


async def convert_video_to_audio_standard(
    video_path: Path, 
    output_path: Path, 
    format: str = "mp3"
) -> Path:
    """
    Standard implementation of video to audio conversion using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file
        format: Audio format (default: mp3)

    Returns:
        Path to the converted audio file
    """
    try:
        logger.info(f"Converting {video_path} to {output_path} (standard implementation)")

        # Build the ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),  # Input file
            "-vn",  # Disable video
            "-acodec", "libmp3lame" if format == "mp3" else format,  # Audio codec
            "-q:a", "2",  # Audio quality (0-9, 0=best)
            "-y",  # Overwrite output file if it exists
            str(output_path)  # Output file
        ]

        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for the process to complete
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")

        logger.info(f"Successfully converted {video_path} to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error converting video to audio: {e}")
        raise


async def convert_video_to_audio_with_profiler(
    video_path: Path, 
    output_path: Path, 
    format: str = "mp3"
) -> Path:
    """
    Memory-profiled implementation of video to audio conversion using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file
        format: Audio format (default: mp3)

    Returns:
        Path to the converted audio file
    """
    try:
        logger.info(f"Converting {video_path} to {output_path} (with memory profiler)")

        # Build the ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),  # Input file
            "-vn",  # Disable video
            "-acodec", "libmp3lame" if format == "mp3" else format,  # Audio codec
            "-q:a", "2",  # Audio quality (0-9, 0=best)
            "-y",  # Overwrite output file if it exists
            str(output_path)  # Output file
        ]

        # Start memory profiling
        memory_stats = []
        process = None

        try:
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Monitor memory usage while the process is running
            while process.poll() is None:
                if process.pid:
                    try:
                        # Get process memory info
                        proc = psutil.Process(process.pid)
                        memory_info = proc.memory_info()

                        # Record memory usage
                        memory_stats.append({
                            'timestamp': time.time(),
                            'rss': memory_info.rss,  # Resident Set Size
                            'vms': memory_info.vms,  # Virtual Memory Size
                        })

                        logger.debug(f"Memory usage: RSS={memory_info.rss / (1024*1024):.2f}MB, VMS={memory_info.vms / (1024*1024):.2f}MB")
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Sleep briefly to avoid excessive CPU usage
                time.sleep(0.1)

            # Get final output
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")

        finally:
            # Ensure process is terminated
            if process and process.poll() is None:
                process.terminate()
                process.wait()

        # Log memory usage statistics
        if memory_stats:
            peak_memory = max(stat['rss'] for stat in memory_stats) / (1024*1024)
            avg_memory = sum(stat['rss'] for stat in memory_stats) / len(memory_stats) / (1024*1024)
            logger.info(f"Memory usage statistics: Peak={peak_memory:.2f}MB, Average={avg_memory:.2f}MB")

        logger.info(f"Successfully converted {video_path} to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error converting video to audio: {e}")
        raise
