from typing import List, Optional

from loguru import logger

from src.utils.text_utils import DEFAULT_BUFFER, DEFAULT_MATCH_CUTOFF, merge_all_chunks
from src.utils.llm_utils import format_text_with_llm


async def merge_transcription_chunks(
    transcription_chunks: List[str],
    buffer: int = DEFAULT_BUFFER,
    match_cutoff: int = DEFAULT_MATCH_CUTOFF,
    username: Optional[str] = None,
) -> str:
    """
    Merge transcription chunks back together, handling overlaps intelligently.

    Args:
        transcription_chunks: List of transcription texts
        buffer: Number of words to consider for overlap detection
        match_cutoff: Minimum length of overlap to consider
        username: Username for quota tracking

    Returns:
        Merged transcription text
    """
    if not transcription_chunks:
        return ""

    if len(transcription_chunks) == 1:
        return transcription_chunks[0]

    logger.info(f"Merging {len(transcription_chunks)} transcription chunks")

    # Use the custom merge_all_chunks function from text_utils
    merged_text = merge_all_chunks(
        transcription_chunks, buffer=buffer, match_cutoff=match_cutoff, logger=logger
    )

    # Format the merged text using LLM to improve readability
    try:
        logger.info("Formatting merged text with LLM")
        formatted_text = await format_text_with_llm(merged_text, username)
        logger.info(f"Formatted text: {len(formatted_text)} characters")
        return formatted_text
    except Exception as e:
        logger.error(f"Error formatting text: {e}")
        return merged_text
