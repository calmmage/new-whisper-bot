import difflib
from typing import List

from loguru import logger


async def merge_transcription_chunks(
    transcription_chunks: List[str],
    overlap_threshold: float = 0.5
) -> str:
    """
    Merge transcription chunks back together, handling overlaps intelligently.
    
    Args:
        transcription_chunks: List of transcription texts
        overlap_threshold: Threshold for determining overlap (0.0-1.0)
        
    Returns:
        Merged transcription text
    """
    if not transcription_chunks:
        return ""
    
    if len(transcription_chunks) == 1:
        return transcription_chunks[0]
    
    logger.info(f"Merging {len(transcription_chunks)} transcription chunks")
    
    # Start with the first chunk
    merged_text = transcription_chunks[0]
    
    # Process each subsequent chunk
    for i in range(1, len(transcription_chunks)):
        current_chunk = transcription_chunks[i]
        
        # Find potential overlap between the end of the merged text and the start of the current chunk
        overlap = find_best_overlap(merged_text, current_chunk, overlap_threshold)
        
        if overlap:
            # Merge with overlap
            overlap_len = len(overlap)
            merged_text_end_pos = merged_text.rfind(overlap)
            
            # If we found a valid position for the overlap
            if merged_text_end_pos != -1:
                # Append only the part of the current chunk that comes after the overlap
                merged_text = merged_text[:merged_text_end_pos] + current_chunk
                logger.info(f"Merged chunk {i} with overlap of {overlap_len} characters")
            else:
                # Fallback: just append with a space
                merged_text += " " + current_chunk
                logger.info(f"Merged chunk {i} with space (no overlap found)")
        else:
            # No significant overlap found, just append with a space
            merged_text += " " + current_chunk
            logger.info(f"Merged chunk {i} with space (no significant overlap)")
    
    logger.info(f"Merged transcription: {len(merged_text)} characters")
    return merged_text


def find_best_overlap(text1: str, text2: str, threshold: float = 0.5, min_overlap_len: int = 10) -> str:
    """
    Find the best overlap between the end of text1 and the start of text2.
    
    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (0.0-1.0)
        min_overlap_len: Minimum length of overlap to consider
        
    Returns:
        The overlapping text, or empty string if no significant overlap found
    """
    # Determine the maximum possible overlap length
    max_overlap_len = min(len(text1), len(text2))
    
    # Start with the longest possible overlap and work down
    for overlap_len in range(max_overlap_len, min_overlap_len - 1, -1):
        # Get the end of text1 and start of text2 with the current overlap length
        text1_end = text1[-overlap_len:]
        text2_start = text2[:overlap_len]
        
        # Calculate similarity using difflib
        similarity = difflib.SequenceMatcher(None, text1_end, text2_start).ratio()
        
        # If similarity is above threshold, we found a good overlap
        if similarity >= threshold:
            # Use the text from text2 as the overlap (could also use text1_end)
            return text2_start
    
    # No significant overlap found
    return ""
