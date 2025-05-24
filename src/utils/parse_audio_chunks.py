import asyncio
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import whisper
from loguru import logger


# Cache for the whisper model to avoid reloading it for each chunk
_whisper_model = None


def get_whisper_model(model_name: str = "base"):
    """
    Get or initialize the Whisper model.
    
    Args:
        model_name: Name of the Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        Whisper model instance
    """
    global _whisper_model
    
    if _whisper_model is None:
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name)
        logger.info(f"Whisper model loaded: {model_name}")
    
    return _whisper_model


async def parse_audio_chunk(
    audio_path: Path,
    model_name: str = "base",
    language: Optional[str] = None,
    device: Optional[str] = None
) -> str:
    """
    Parse a single audio chunk using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_name: Name of the Whisper model to use (tiny, base, small, medium, large)
        language: Language code (optional, auto-detected if None)
        device: Device to use for inference (cpu, cuda, etc.)
        
    Returns:
        Transcription text
    """
    # Determine the device to use
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Transcribing {audio_path} using Whisper model {model_name} on {device}")
    
    # Get the Whisper model
    model = get_whisper_model(model_name)
    
    try:
        # Run the transcription in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                str(audio_path),
                language=language,
                fp16=(device == "cuda")
            )
        )
        
        # Extract the transcription text
        transcription = result["text"].strip()
        logger.info(f"Transcription complete for {audio_path}: {len(transcription)} characters")
        
        return transcription
        
    except Exception as e:
        logger.error(f"Error transcribing audio chunk {audio_path}: {e}")
        raise


async def parse_audio_chunks(
    audio_chunks: List[Path],
    model_name: str = "base",
    language: Optional[str] = None,
    device: Optional[str] = None,
    max_concurrent: int = 1
) -> List[str]:
    """
    Parse multiple audio chunks using Whisper.
    
    Args:
        audio_chunks: List of paths to audio files
        model_name: Name of the Whisper model to use (tiny, base, small, medium, large)
        language: Language code (optional, auto-detected if None)
        device: Device to use for inference (cpu, cuda, etc.)
        max_concurrent: Maximum number of concurrent transcriptions
        
    Returns:
        List of transcription texts
    """
    logger.info(f"Transcribing {len(audio_chunks)} audio chunks")
    
    # Process chunks in batches to limit memory usage
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_chunk(chunk_path):
        async with semaphore:
            return await parse_audio_chunk(chunk_path, model_name, language, device)
    
    # Create tasks for all chunks
    tasks = [process_chunk(chunk) for chunk in audio_chunks]
    
    # Wait for all tasks to complete
    transcriptions = await asyncio.gather(*tasks)
    
    logger.info(f"Transcription complete for all {len(audio_chunks)} chunks")
    return transcriptions
