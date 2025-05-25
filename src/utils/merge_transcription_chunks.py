from typing import List, Optional

from botspot.llm_provider import aquery_llm_text
from loguru import logger

from src.utils.text_utils import merge_all_chunks


async def merge_transcription_chunks(
    transcription_chunks: List[str],
    buffer: int = 25,
    match_cutoff: int = 15,
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


async def format_text_with_llm(
    text: str, username: Optional[str] = None, model: str = "gpt-4-1106-preview"
) -> str:
    """
    Format text with LLM to improve readability.

    Args:
        text: Text to format
        username: Username for quota tracking
        model: LLM model to use

    Returns:
        Formatted text
    """
    system_prompt = """
    You're text formatting assistant. Your goal is:
    - Add rich punctuation - new lines, quotes, dots and commas where appropriate
    - Break the text into paragraphs using double new lines
    - Keep the original text word-to-word, with only minor changes where absolutely necessary
    - Fix grammar and typos only when they significantly impact readability
    """

    prompt = f"""
    Please format the following transcription text to improve readability:

    {text}
    """

    formatted_text = await aquery_llm_text(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        user=username,
        temperature=0.3,
        max_tokens=4096,
    )

    return formatted_text
