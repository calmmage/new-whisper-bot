import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import psutil
from loguru import logger


async def cut_audio_into_pieces(
    audio_path: Path,
    output_dir: Optional[Path] = None,
    chunk_duration: int = 600,  # 10 minutes in seconds
    overlap_duration: int = 30,  # 30 seconds overlap
    format: str = "mp3",
    use_memory_profiler: bool = False
) -> List[Path]:
    """
    Cut audio file into smaller pieces with optional overlap.

    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the audio pieces (default: same as audio_path)
        chunk_duration: Duration of each chunk in seconds (default: 600 seconds = 10 minutes)
        overlap_duration: Overlap duration between chunks in seconds (default: 30 seconds)
        format: Audio format for the output chunks (default: mp3)
        use_memory_profiler: Whether to use memory profiler implementation

    Returns:
        List of paths to the audio pieces
    """
    # If output_dir is not provided, use the same directory as the audio file
    if output_dir is None:
        output_dir = audio_path.parent / f"{audio_path.stem}_chunks"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the duration of the audio file using ffprobe
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"FFprobe failed: {error_msg}")

        duration = float(stdout.decode().strip())
        logger.info(f"Audio duration: {duration} seconds")

    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        raise

    # Calculate the number of chunks
    effective_chunk_duration = chunk_duration - overlap_duration
    num_chunks = max(1, int((duration - overlap_duration) / effective_chunk_duration) + 1)

    logger.info(f"Cutting audio into {num_chunks} chunks of {chunk_duration} seconds with {overlap_duration} seconds overlap")

    # Choose the appropriate implementation based on the flag
    if use_memory_profiler:
        return await cut_audio_into_pieces_with_profiler(
            audio_path, output_dir, duration, num_chunks, 
            chunk_duration, overlap_duration, format
        )
    else:
        return await cut_audio_into_pieces_standard(
            audio_path, output_dir, duration, num_chunks, 
            chunk_duration, overlap_duration, format
        )


async def cut_audio_into_pieces_standard(
    audio_path: Path,
    output_dir: Path,
    duration: float,
    num_chunks: int,
    chunk_duration: int = 600,
    overlap_duration: int = 30,
    format: str = "mp3"
) -> List[Path]:
    """
    Standard implementation of cutting audio into pieces using ffmpeg.

    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the audio pieces
        duration: Duration of the audio file in seconds
        num_chunks: Number of chunks to create
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap duration between chunks in seconds
        format: Audio format for the output chunks

    Returns:
        List of paths to the audio pieces
    """
    logger.info(f"Cutting audio into pieces (standard implementation)")

    effective_chunk_duration = chunk_duration - overlap_duration
    chunk_paths = []

    for i in range(num_chunks):
        # Calculate start and end times for this chunk
        start_time = i * effective_chunk_duration
        end_time = min(start_time + chunk_duration, duration)

        # Skip if we've reached the end of the audio
        if start_time >= duration:
            break

        # Create output path for this chunk
        chunk_path = output_dir / f"{audio_path.stem}_chunk_{i+1:03d}.{format}"

        try:
            # Build the ffmpeg command
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),  # Input file
                "-ss", str(start_time),  # Start time
                "-to", str(end_time),  # End time
                "-c:a", "libmp3lame" if format == "mp3" else format,  # Audio codec
                "-q:a", "2",  # Audio quality (0-9, 0=best)
                "-y",  # Overwrite output file if it exists
                str(chunk_path)  # Output file
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
                raise RuntimeError(f"FFmpeg cutting failed: {error_msg}")

            logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            chunk_paths.append(chunk_path)

        except Exception as e:
            logger.error(f"Error cutting audio chunk {i+1}: {e}")
            raise

    return chunk_paths


async def cut_audio_into_pieces_with_profiler(
    audio_path: Path,
    output_dir: Path,
    duration: float,
    num_chunks: int,
    chunk_duration: int = 600,
    overlap_duration: int = 30,
    format: str = "mp3"
) -> List[Path]:
    """
    Memory-profiled implementation of cutting audio into pieces using ffmpeg.

    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the audio pieces
        duration: Duration of the audio file in seconds
        num_chunks: Number of chunks to create
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap duration between chunks in seconds
        format: Audio format for the output chunks

    Returns:
        List of paths to the audio pieces
    """
    logger.info(f"Cutting audio into pieces (with memory profiler)")

    effective_chunk_duration = chunk_duration - overlap_duration
    chunk_paths = []
    all_memory_stats = []

    for i in range(num_chunks):
        # Calculate start and end times for this chunk
        start_time = i * effective_chunk_duration
        end_time = min(start_time + chunk_duration, duration)

        # Skip if we've reached the end of the audio
        if start_time >= duration:
            break

        # Create output path for this chunk
        chunk_path = output_dir / f"{audio_path.stem}_chunk_{i+1:03d}.{format}"

        try:
            # Build the ffmpeg command
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),  # Input file
                "-ss", str(start_time),  # Start time
                "-to", str(end_time),  # End time
                "-c:a", "libmp3lame" if format == "mp3" else format,  # Audio codec
                "-q:a", "2",  # Audio quality (0-9, 0=best)
                "-y",  # Overwrite output file if it exists
                str(chunk_path)  # Output file
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
                                'chunk': i+1
                            })

                            logger.debug(f"Chunk {i+1} memory usage: RSS={memory_info.rss / (1024*1024):.2f}MB, VMS={memory_info.vms / (1024*1024):.2f}MB")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    # Sleep briefly to avoid excessive CPU usage
                    time.sleep(0.1)

                # Get final output
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    raise RuntimeError(f"FFmpeg cutting failed: {error_msg}")

            finally:
                # Ensure process is terminated
                if process and process.poll() is None:
                    process.terminate()
                    process.wait()

            # Add memory stats to overall stats
            all_memory_stats.extend(memory_stats)

            # Log memory usage statistics for this chunk
            if memory_stats:
                peak_memory = max(stat['rss'] for stat in memory_stats) / (1024*1024)
                avg_memory = sum(stat['rss'] for stat in memory_stats) / len(memory_stats) / (1024*1024)
                logger.info(f"Chunk {i+1} memory usage: Peak={peak_memory:.2f}MB, Average={avg_memory:.2f}MB")

            logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            chunk_paths.append(chunk_path)

        except Exception as e:
            logger.error(f"Error cutting audio chunk {i+1}: {e}")
            raise

    # Log overall memory usage statistics
    if all_memory_stats:
        peak_memory = max(stat['rss'] for stat in all_memory_stats) / (1024*1024)
        avg_memory = sum(stat['rss'] for stat in all_memory_stats) / len(all_memory_stats) / (1024*1024)
        logger.info(f"Overall memory usage statistics: Peak={peak_memory:.2f}MB, Average={avg_memory:.2f}MB")

    return chunk_paths
