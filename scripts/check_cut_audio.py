#!/usr/bin/env python3
"""
Test script for the cut_audio_into_pieces utility.

This script tests both the standard and memory-profiled implementations
of the cut_audio_into_pieces function.

Usage:
    python scripts/check_cut_audio.py
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.cut_audio import cut_audio_into_pieces


def get_audio_duration(audio_path: Path) -> float:
    """
    Get the duration of an audio file using ffprobe.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration of the audio file in seconds
    """
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
    
    return float(stdout.decode().strip())


async def check_cut_audio(
    audio_path: Path,
    output_dir: Path,
    chunk_duration: int = 60,  # 1 minute for testing
    overlap_duration: int = 5,  # 5 seconds overlap
    use_memory_profiler: bool = False
):
    """
    Test the cut_audio_into_pieces function.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the output audio pieces
        chunk_duration: Duration of each chunk in seconds
        overlap_duration: Overlap duration between chunks in seconds
        use_memory_profiler: Whether to use the memory-profiled implementation
    """
    logger.info(f"Testing cut_audio_into_pieces with {'memory profiler' if use_memory_profiler else 'standard'} implementation")
    
    # Create a specific output directory for this test
    test_output_dir = output_dir / f"{audio_path.stem}_{'profiled' if use_memory_profiler else 'standard'}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the original audio duration
    original_duration = get_audio_duration(audio_path)
    logger.info(f"Original audio duration: {original_duration:.2f} seconds")
    
    # Calculate expected number of chunks
    effective_chunk_duration = chunk_duration - overlap_duration
    expected_chunks = max(1, int((original_duration - overlap_duration) / effective_chunk_duration) + 1)
    logger.info(f"Expected number of chunks: {expected_chunks}")
    
    # Cut the audio into pieces
    try:
        start_time = asyncio.get_event_loop().time()
        audio_pieces = await cut_audio_into_pieces(
            audio_path=audio_path,
            output_dir=test_output_dir,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            format="mp3",
            use_memory_profiler=use_memory_profiler
        )
        end_time = asyncio.get_event_loop().time()
        
        # Sanity checks
        
        # Check number of pieces
        if len(audio_pieces) != expected_chunks:
            logger.warning(f"Number of pieces ({len(audio_pieces)}) doesn't match expected ({expected_chunks})")
        
        # Check if all pieces exist
        for piece in audio_pieces:
            if not piece.exists():
                logger.error(f"Output file {piece} does not exist")
                return False
            
            # Check file size
            file_size = piece.stat().st_size
            if file_size == 0:
                logger.error(f"Output file {piece} is empty")
                return False
            
            # Check duration
            try:
                duration = get_audio_duration(piece)
                logger.info(f"Piece {piece.name} duration: {duration:.2f} seconds")
                
                # Check if duration is within expected range
                if duration > chunk_duration + 1:  # Allow 1 second tolerance
                    logger.warning(f"Piece {piece.name} duration ({duration:.2f}s) exceeds chunk duration ({chunk_duration}s)")
            except Exception as e:
                logger.error(f"Error getting duration for {piece}: {e}")
        
        # Log total size of all pieces
        total_size = sum(piece.stat().st_size for piece in audio_pieces)
        logger.info(f"Total size of all pieces: {total_size / (1024*1024):.2f} MB")
        logger.info(f"Cutting time: {end_time - start_time:.2f} seconds")
        
        return True
    
    except Exception as e:
        logger.error(f"Error cutting audio into pieces: {e}")
        return False


async def main():
    """Main function."""
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("scripts/check_cut_audio.log", level="DEBUG", rotation="10 MB")
    
    # Define paths
    sample_dir = Path(__file__).parent / "sample_data_files" / "sample_data"
    output_dir = Path(__file__).parent / "output" / "cut_audio"
    
    # Find audio files in the sample directory
    audio_files = [f for f in sample_dir.glob("*.m4a")] + [f for f in sample_dir.glob("*.mp3")] + [f for f in sample_dir.glob("*.ogg")]
    audio_files +=[
        Path('/Users/petrlavrov/work/experiments/new-whisper-bot/scripts/output/convert_video/GMT20250321-200734_Recording_1920x1170_standard.mp3')
    ]
    
    if not audio_files:
        logger.error(f"No audio files found in {sample_dir}")
        return
    
    # Test with each audio file
    for audio_file in audio_files:
        logger.info(f"Testing with {audio_file}")
        
        # Test standard implementation
        success_standard = await check_cut_audio(
            audio_path=audio_file,
            output_dir=output_dir,
            use_memory_profiler=False
        )
        
        # Test memory-profiled implementation
        success_profiled = await check_cut_audio(
            audio_path=audio_file,
            output_dir=output_dir,
            use_memory_profiler=True
        )
        
        if success_standard and success_profiled:
            logger.info(f"All tests passed for {audio_file}")
        else:
            logger.error(f"Some tests failed for {audio_file}")


if __name__ == "__main__":
    # Create the output directory
    output_dir = Path(__file__).parent / "output" / "cut_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the main function
    asyncio.run(main())