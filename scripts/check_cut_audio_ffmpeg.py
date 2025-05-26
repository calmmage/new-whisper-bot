#!/usr/bin/env python3
"""
Test script for the cut_audio_ffmpeg utility.

Usage:
    python scripts/check_cut_audio_ffmpeg.py
"""

import asyncio
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.new.utils.cut_audio_ffmpeg import cut_audio_ffmpeg


def get_audio_duration(audio_path: Path) -> float:
    """Get the duration of an audio file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        raise RuntimeError(f"FFprobe failed: {error_msg}")

    return float(stdout.decode().strip())


async def test_cut_audio_ffmpeg(audio_path: Path, output_dir: Path, num_parts: int = 3):
    """Test the cut_audio_ffmpeg function."""
    logger.info(f"Testing cut_audio_ffmpeg with {num_parts} parts")

    # Create output directory and clean it
    test_output_dir = output_dir / f"{audio_path.stem}_ffmpeg_test"
    if test_output_dir.exists():
        # Clean up any existing files
        for file in test_output_dir.glob("*"):
            file.unlink()
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Get original audio duration
    original_duration = get_audio_duration(audio_path)
    logger.info(f"Original audio duration: {original_duration:.2f} seconds")

    expected_part_duration = original_duration / num_parts
    logger.info(f"Expected part duration: {expected_part_duration:.2f} seconds")

    try:
        start_time = asyncio.get_event_loop().time()
        
        # Cut the audio
        parts_created = await cut_audio_ffmpeg(
            audio_file=audio_path,
            num_parts=num_parts,
            target_dir=test_output_dir
        )
        
        end_time = asyncio.get_event_loop().time()
        
        logger.info(f"Created {parts_created} parts in {end_time - start_time:.2f} seconds")

        # Check the created files
        created_files = list(test_output_dir.glob(f"{audio_path.stem}_part_*{audio_path.suffix}"))
        created_files.sort()

        if len(created_files) != parts_created:
            logger.error(f"Mismatch: function returned {parts_created} but found {len(created_files)} files")
            return False

        total_duration = 0
        for i, part_file in enumerate(created_files, 1):
            if not part_file.exists():
                logger.error(f"Part file {part_file} does not exist")
                return False

            file_size = part_file.stat().st_size
            if file_size == 0:
                logger.error(f"Part file {part_file} is empty")
                return False

            try:
                duration = get_audio_duration(part_file)
                total_duration += duration
                logger.info(f"Part {i}: {part_file.name} - {duration:.2f}s ({file_size / (1024*1024):.2f} MB)")
            except Exception as e:
                logger.error(f"Error getting duration for {part_file}: {e}")
                return False

        logger.info(f"Total duration of parts: {total_duration:.2f}s (original: {original_duration:.2f}s)")
        
        # Check if total duration is close to original (within 1 second tolerance)
        if abs(total_duration - original_duration) > 1.0:
            logger.warning(f"Duration mismatch: {abs(total_duration - original_duration):.2f}s difference")

        return True

    except Exception as e:
        logger.error(f"Error cutting audio: {e}")
        return False


async def main():
    """Main function."""
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("scripts/check_cut_audio_ffmpeg.log", level="DEBUG", rotation="10 MB")

    # Define paths
    sample_dir = Path(__file__).parent / "sample_data_files" / "sample_data"
    output_dir = Path(__file__).parent / "output" / "cut_audio_ffmpeg"

    # Find audio files
    audio_files = (
        list(sample_dir.glob("*.m4a")) +
        list(sample_dir.glob("*.mp3")) +
        list(sample_dir.glob("*.ogg"))
    )
    
    # Add the specific file from the old script
    specific_file = Path("/Users/petrlavrov/work/experiments/new-whisper-bot/scripts/output/convert_video/GMT20250321-200734_Recording_1920x1170_standard.mp3")
    if specific_file.exists():
        audio_files.append(specific_file)

    if not audio_files:
        logger.error(f"No audio files found in {sample_dir}")
        return

    # Test with different numbers of parts
    test_cases = [2, 3, 5]

    for audio_file in audio_files:
        logger.info(f"Testing with {audio_file}")
        
        for num_parts in test_cases:
            success = await test_cut_audio_ffmpeg(
                audio_path=audio_file,
                output_dir=output_dir,
                num_parts=num_parts
            )
            
            if success:
                logger.info(f"✓ Test passed for {audio_file.name} with {num_parts} parts")
            else:
                logger.error(f"✗ Test failed for {audio_file.name} with {num_parts} parts")


if __name__ == "__main__":
    # Create the output directory
    output_dir = Path(__file__).parent / "output" / "cut_audio_ffmpeg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the main function
    asyncio.run(main()) 