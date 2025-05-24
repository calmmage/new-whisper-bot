import os
from pathlib import Path
from typing import Optional

from botspot.llm_provider import aquery_llm_text
from loguru import logger


async def create_summary(
    transcription: str,
    max_length: int = 1000,
    username: Optional[str] = None,
    model: str = "gpt-4-1106-preview"
) -> str:
    """
    Create a summary of the transcription using botspot's llm_provider.

    Args:
        transcription: The transcription text to summarize
        max_length: Maximum length of the summary in characters
        username: Username for quota tracking
        model: LLM model to use (default: gpt-4-1106-preview)

    Returns:
        Summary text
    """
    if not transcription:
        return "No transcription provided."

    logger.info(f"Creating summary of transcription ({len(transcription)} characters)")

    try:
        # Load the system prompt from file
        system_prompt_path = Path(__file__).parent.parent.parent / "dev" / "summary_system_prompt.txt"
        if system_prompt_path.exists():
            with open(system_prompt_path, "r") as f:
                system_prompt = f.read()
        else:
            system_prompt = (
                "Summarize key points from the conversation and exportable artifacts\n"
                "explicitly make a list of \n"
                "- action points for specific people\n"
                "- general action points - cued by the explicit phrases \"we should do this\" etc.\n"
                "- other meaningful groups \n\n"
                "Format should be with specific simple bullet points\n"
                "\"\"\"\n"
                "Artifact name\n"
                "- point 1\n"
                "- point 2\n\n"
                "Group 2\n"
                "- point 3\n"
                "- point 4\n"
                "\"\"\"\n\n"
                "Summary should be in the language of the original"
            )

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

        # Call LLM using botspot's llm_provider
        summary = await aquery_llm_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            user=username,
            temperature=0.3,
            max_tokens=1024
        )

        logger.info(f"Summary created: {len(summary)} characters")
        return summary

    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        # Provide a fallback summary in case of error
        return f"Error creating summary: {str(e)}\n\nOriginal transcription: {transcription[:200]}..."
