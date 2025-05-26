from pathlib import Path
import asyncio
from typing import Optional


async def cut_audio_ffmpeg(audio_file: Path, num_parts: Optional[int] = None, part_duration: Optional[float] = None, target_dir: Optional[Path] = None):
    assert num_parts is not None or part_duration is not None

    if target_dir is None:
        target_dir = audio_file.parent / audio_file.stem
    
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True)
    
    if num_parts:
        # Get audio duration to calculate segment time
        duration_cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(audio_file)
        ]
        result = await asyncio.create_subprocess_exec(
            *duration_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get audio duration: {stderr.decode()}")
        
        total_duration = float(stdout.decode().strip())
        segment_time = total_duration / num_parts
    else:
        segment_time = part_duration
    
    # Use ffmpeg segment to cut in one command
    output_pattern = target_dir / f"{audio_file.stem}_part_%03d{audio_file.suffix}"
    
    segment_cmd = [
        "ffmpeg", "-i", str(audio_file),
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-c", "copy",
        "-y",  # overwrite output files
        str(output_pattern)
    ]
    
    result = await asyncio.create_subprocess_exec(
        *segment_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to segment audio: {stderr.decode()}")
    
    # Count the created files
    created_files = list(target_dir.glob(f"{audio_file.stem}_part_*{audio_file.suffix}"))
    return len(created_files)