#!/usr/bin/env python3
"""
Test script for the convert_video_to_audio utility.

This script tests both the standard and memory-profiled implementations
of the convert_video_to_audio function.

Usage:
    python scripts/check_convert_video.py
"""

import asyncio
import sys
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.new.utils.convert_to_mp3_ffmpeg import convert_to_mp3_ffmpeg


async def check_convert_video(
    video_path: Path, output_dir: Path, use_memory_profiler: bool = False
):
    """
    Test the convert_video_to_audio function.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the output audio file
        use_memory_profiler: Whether to use the memory-profiled implementation
    """
    logger.info(
        f"Testing convert_video_to_audio with {'memory profiler' if use_memory_profiler else 'standard'} implementation"
    )

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the output path
    output_path = (
        output_dir
        / f"{video_path.stem}_{'profiled' if use_memory_profiler else 'standard'}.mp3"
    )

    # Convert the video to audio
    try:
        start_time = asyncio.get_event_loop().time()
        audio_path = await convert_to_mp3_ffmpeg(
            source_path=video_path,
            output_path=output_path,
            format="mp3",
            use_memory_profiler=use_memory_profiler,
        )
        end_time = asyncio.get_event_loop().time()

        # Sanity checks
        if not audio_path.exists():
            logger.error(f"Output file {audio_path} does not exist")
            return False

        # Check file size
        file_size = audio_path.stat().st_size
        if file_size == 0:
            logger.error(f"Output file {audio_path} is empty")
            return False

        logger.info(f"Successfully converted {video_path} to {audio_path}")
        logger.info(f"File size: {file_size / (1024 * 1024):.2f} MB")
        logger.info(f"Conversion time: {end_time - start_time:.2f} seconds")

        return True

    except Exception as e:
        logger.error(f"Error converting video to audio: {e}")
        return False


async def main():
    """Main function."""
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("scripts/check_convert_video.log", level="DEBUG", rotation="10 MB")

    # Define paths
    sample_dir = Path(__file__).parent / "sample_data_files" / "sample_data"
    output_dir = Path(__file__).parent / "output" / "convert_video"

    # Find video files in the sample directory
    source_files = [f for f in sample_dir.glob("*.mp4")]
    source_files += [f for f in sample_dir.glob("*.ogg")]
    source_files += [
        Path("/Users/petrlavrov/Downloads/GMT20250321-200734_Recording_1920x1170.mp4")
    ]

    if not source_files:
        logger.error(f"No video files found in {sample_dir}")
        return

    # Test with each video file
    for video_file in source_files:
        logger.info(f"Testing with {video_file}")

        # Test standard implementation
        success_standard = await check_convert_video(
            video_path=video_file, output_dir=output_dir, use_memory_profiler=False
        )

        # Test memory-profiled implementation
        success_profiled = await check_convert_video(
            video_path=video_file, output_dir=output_dir, use_memory_profiler=True
        )

        if success_standard and success_profiled:
            logger.info(f"All tests passed for {video_file}")
        else:
            logger.error(f"Some tests failed for {video_file}")


if __name__ == "__main__":
    # Create the output directory
    output_dir = Path(__file__).parent / "output" / "convert_video"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the main function
    asyncio.run(main())
