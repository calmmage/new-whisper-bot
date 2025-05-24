#!/usr/bin/env python3
"""
Test script for the parse_audio_chunks utility.

This script tests the OpenAI Whisper API implementation for transcribing audio files.

Usage:
    python scripts/check_parse_audio.py
"""

import asyncio
import os
import sys
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.parse_audio_chunks import parse_audio_chunk, parse_audio_chunks


async def check_parse_audio_chunk(
    audio_path: Path, output_dir: Path, api_key: str = None
):
    """
    Test the parse_audio_chunk function.

    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the output transcription
        api_key: OpenAI API key (optional)
    """
    logger.info(f"Testing parse_audio_chunk with {audio_path}")

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the output path for the transcription
    output_path = output_dir / f"{audio_path.stem}_transcription.txt"

    # Parse the audio
    try:
        start_time = asyncio.get_event_loop().time()
        transcription = await parse_audio_chunk(
            audio_path=audio_path, model_name="whisper-1", api_key=api_key
        )
        end_time = asyncio.get_event_loop().time()

        # Save the transcription to a file
        with open(output_path, "w") as f:
            f.write(transcription)

        # Sanity checks
        if not output_path.exists():
            logger.error(f"Output file {output_path} does not exist")
            return False

        # Check transcription length
        if len(transcription) == 0:
            logger.error("Transcription is empty")
            return False

        logger.info(f"Successfully transcribed {audio_path}")
        logger.info(f"Transcription length: {len(transcription)} characters")
        logger.info(f"Transcription time: {end_time - start_time:.2f} seconds")
        logger.info(f"Transcription saved to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Error parsing audio: {e}")
        return False


async def check_parse_audio_chunks(
    audio_paths: list[Path], output_dir: Path, api_key: str = None
):
    """
    Test the parse_audio_chunks function.

    Args:
        audio_paths: List of paths to audio files
        output_dir: Directory to save the output transcriptions
        api_key: OpenAI API key (optional)
    """
    logger.info(f"Testing parse_audio_chunks with {len(audio_paths)} audio files")

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set the output path for the combined transcription
    output_path = output_dir / "combined_transcription.txt"

    # Parse the audio chunks
    try:
        start_time = asyncio.get_event_loop().time()
        transcriptions = await parse_audio_chunks(
            audio_chunks=audio_paths,
            model_name="whisper-1",
            api_key=api_key,
            max_concurrent=2,  # Limit concurrent API calls for testing
        )
        end_time = asyncio.get_event_loop().time()

        # Save the individual transcriptions to files
        for i, transcription in enumerate(transcriptions):
            chunk_output_path = output_dir / f"chunk_{i + 1:03d}_transcription.txt"
            with open(chunk_output_path, "w") as f:
                f.write(transcription)

        # Save the combined transcription to a file
        combined_transcription = "\n\n".join(transcriptions)
        with open(output_path, "w") as f:
            f.write(combined_transcription)

        # Sanity checks
        if not output_path.exists():
            logger.error(f"Output file {output_path} does not exist")
            return False

        # Check number of transcriptions
        if len(transcriptions) != len(audio_paths):
            logger.error(
                f"Number of transcriptions ({len(transcriptions)}) doesn't match number of audio files ({len(audio_paths)})"
            )
            return False

        # Check transcription lengths
        for i, transcription in enumerate(transcriptions):
            if len(transcription) == 0:
                logger.error(f"Transcription {i + 1} is empty")
                return False
            logger.info(
                f"Transcription {i + 1} length: {len(transcription)} characters"
            )

        logger.info(f"Successfully transcribed {len(audio_paths)} audio files")
        logger.info(f"Total transcription time: {end_time - start_time:.2f} seconds")
        logger.info(f"Combined transcription saved to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Error parsing audio chunks: {e}")
        return False


async def main():
    """Main function."""
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("scripts/check_parse_audio.log", level="DEBUG", rotation="10 MB")

    # Get OpenAI API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return

    # Define paths
    sample_dir = Path(__file__).parent / "sample_data_files" / "sample_data"
    output_dir = Path(__file__).parent / "output" / "parse_audio"

    # Find audio files in the sample directory
    audio_files = (
        [f for f in sample_dir.glob("*.m4a")]
        + [f for f in sample_dir.glob("*.mp3")]
        + [f for f in sample_dir.glob("*.ogg")]
    )

    if not audio_files:
        logger.error(f"No audio files found in {sample_dir}")
        return

    # Test with a single audio file first
    logger.info("Testing single audio file transcription")
    single_audio_file = audio_files[0]
    single_output_dir = output_dir / "single"

    success_single = await check_parse_audio_chunk(
        audio_path=single_audio_file, output_dir=single_output_dir, api_key=api_key
    )

    if success_single:
        logger.info(f"Single audio file test passed for {single_audio_file}")
    else:
        logger.error(f"Single audio file test failed for {single_audio_file}")

    # Test with multiple audio files (use a small subset to avoid excessive API usage)
    logger.info("Testing multiple audio files transcription")
    # Use a small voice file for testing multiple chunks
    voice_files = [f for f in sample_dir.glob("*voice*.ogg")]
    if voice_files:
        # Use voice files for testing multiple chunks
        test_files = voice_files
    else:
        # If no voice files, use the first audio file
        test_files = [audio_files[0]]

    multi_output_dir = output_dir / "multiple"

    success_multi = await check_parse_audio_chunks(
        audio_paths=test_files, output_dir=multi_output_dir, api_key=api_key
    )

    if success_multi:
        logger.info("Multiple audio files test passed")
    else:
        logger.error("Multiple audio files test failed")


if __name__ == "__main__":
    # Create the output directory
    output_dir = Path(__file__).parent / "output" / "parse_audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the main function
    asyncio.run(main())
