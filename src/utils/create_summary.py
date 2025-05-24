import os
from typing import Optional

import anthropic
from loguru import logger


async def create_summary(
    transcription: str,
    max_length: int = 1000,
    api_key: Optional[str] = None,
    model: str = "claude-3-haiku-20240307"
) -> str:
    """
    Create a summary of the transcription using Claude.
    
    Args:
        transcription: The transcription text to summarize
        max_length: Maximum length of the summary in characters
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
        model: Claude model to use
        
    Returns:
        Summary text
    """
    if not transcription:
        return "No transcription provided."
    
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    logger.info(f"Creating summary of transcription ({len(transcription)} characters)")
    
    try:
        # Initialize Claude client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Create the prompt
        prompt = f"""
        Please create a concise summary of the following transcription. 
        The summary should:
        1. Capture the main points and key information
        2. Be well-structured with bullet points for main topics
        3. Be no longer than {max_length} characters
        4. Include a very brief 1-2 sentence overview at the beginning

        Here is the transcription:
        {transcription}
        """
        
        # Call Claude API
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.3,
            system="You are a helpful assistant that creates concise, accurate summaries.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the summary
        summary = response.content[0].text
        
        logger.info(f"Summary created: {len(summary)} characters")
        return summary
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        # Provide a fallback summary in case of error
        return f"Error creating summary: {str(e)}\n\nOriginal transcription: {transcription[:200]}..."
