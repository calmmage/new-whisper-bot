import asyncio
import os
from pathlib import Path
from typing import List, Optional

import openai
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
    ),
    reraise=True,
)
async def parse_audio_chunk(
    audio_path: Path,
    model_name: str = "whisper-1",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Parse a single audio chunk using OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file
        model_name: Name of the Whisper model to use (whisper-1)
        language: Language code (optional, auto-detected if None)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)

    Returns:
        Transcription text
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    logger.info(f"Transcribing {audio_path} using OpenAI Whisper API")

    # Initialize OpenAI client
    client = openai.AsyncOpenAI(api_key=api_key)

    try:
        # Open the audio file
        with open(audio_path, "rb") as audio_file:
            # Call the OpenAI API
            options = {"response_format": "text"}
            if language:
                options["language"] = language

            response = await client.audio.transcriptions.create(
                file=audio_file, model=model_name, **options
            )

            # Extract the transcription text
            transcription = response
            logger.info(
                f"Transcription complete for {audio_path}: {len(transcription)} characters"
            )

            return transcription

    except Exception as e:
        logger.error(f"Error transcribing audio chunk {audio_path}: {e}")
        raise


async def parse_audio_chunks(
    audio_chunks: List[Path],
    model_name: str = "whisper-1",
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    max_concurrent: int = 3,
) -> List[str]:
    """
    Parse multiple audio chunks using OpenAI Whisper API.

    Args:
        audio_chunks: List of paths to audio files
        model_name: Name of the Whisper model to use (whisper-1)
        language: Language code (optional, auto-detected if None)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        max_concurrent: Maximum number of concurrent transcriptions

    Returns:
        List of transcription texts
    """
    logger.info(f"Transcribing {len(audio_chunks)} audio chunks")

    # Process chunks in batches to limit API usage
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_chunk(chunk_path):
        async with semaphore:
            # Add a small delay between API calls to avoid rate limits
            await asyncio.sleep(0.5)
            return await parse_audio_chunk(chunk_path, model_name, language, api_key)

    # Create tasks for all chunks
    tasks = [process_chunk(chunk) for chunk in audio_chunks]

    # Wait for all tasks to complete
    transcriptions = await asyncio.gather(*tasks)

    logger.info(f"Transcription complete for all {len(audio_chunks)} chunks")
    return transcriptions
