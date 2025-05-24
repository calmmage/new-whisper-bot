import os
import subprocess
from pathlib import Path
from typing import List, Optional

from loguru import logger


async def cut_audio_into_pieces(
    audio_path: Path,
    output_dir: Optional[Path] = None,
    chunk_duration: int = 600,  # 10 minutes in seconds
    overlap_duration: int = 30,  # 30 seconds overlap
    format: str = "mp3"
) -> List[Path]:
    """
    Cut audio file into smaller pieces with optional overlap.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the audio pieces (default: same as audio_path)
        chunk_duration: Duration of each chunk in seconds (default: 600 seconds = 10 minutes)
        overlap_duration: Overlap duration between chunks in seconds (default: 30 seconds)
        format: Audio format for the output chunks (default: mp3)
        
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
    
    # Cut the audio file into chunks
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
