import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


async def convert_video_to_audio(
    video_path: Path, 
    output_path: Optional[Path] = None, 
    format: str = "mp3"
) -> Path:
    """
    Convert video file to audio using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file (optional)
        format: Audio format (default: mp3)
        
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
    
    # Convert video to audio using ffmpeg
    try:
        logger.info(f"Converting {video_path} to {output_path}")
        
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
