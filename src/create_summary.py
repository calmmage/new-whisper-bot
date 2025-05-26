from textwrap import dedent
from typing import Optional

from botspot.llm_provider import aquery_llm_text
from loguru import logger


async def create_summary(
    transcription: str,
    # max_length: int = 1000,
    username: Optional[str] = None,
    model: str = "claude-4",
    # option 1 - claude 3.5, option 2 - claude 3.7, option 3 - claude 4
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """
    Create a summary of the transcription using botspot's llm_provider.

    Args:
        transcription: The transcription text to summarize
        max_tokens: Maximum length of the summary in tokens
        username: Username for quota tracking
        model: LLM model to use (default: gpt-4-1106-preview)

    Returns:
        Summary text
    """
    if not transcription:
        return "No transcription provided."

    logger.info(f"Creating summary of transcription ({len(transcription)} characters)")

    # Load the system prompt from file
    # system_prompt_path = (
    #     Path(__file__).parent.parent.parent / "dev" / "summary_system_prompt.txt"
    # )
    # if system_prompt_path.exists():
    #     with open(system_prompt_path) as f:
    #         system_prompt = f.read()
    # else:
    system_message = dedent("""
        Summarize key points from the conversation and exportable artifacts
        explicitly make a list of 
        - action points for specific people
        - general action points - cued by the explicit phrases "we should do this" etc.
        - other meaningful groups 
        Format should be with specific simple bullet points
        '''
        Artifact name
        - point 1
        - point 2
        Group 2
        - point 3
        - point 4
        '''
        Summary should be in the language of the original
        """)

    # Create the prompt
    # prompt = f"""
    # Please create a concise summary of the following transcription.
    # The summary should:
    # 1. Capture the main points and key information
    # 2. Be well-structured with bullet points for main topics
    # 3. Be no longer than {max_length} characters
    # 4. Include a very brief 1-2 sentence overview at the beginning
    #
    # Here is the transcription:
    # {transcription}
    # """

    # Call LLM using botspot's llm_provider
    summary = await aquery_llm_text(
        # prompt=prompt,
        prompt=transcription,
        system_message=system_message,
        model=model,
        user=username,
        # temperature=0.3,
        max_tokens=max_tokens,
        **kwargs,
    )

    logger.info(f"Summary created: {len(summary)} characters")
    return summary
