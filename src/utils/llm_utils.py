from typing import Optional

from botspot.llm_provider import aquery_llm_text


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
    
    Example:
    Input: "this isa test. ohh. um.. it should be 1formatted corr ectly. phew"
    Output: "This is a test. It should be formatted correctly."
    """

    formatted_text = await aquery_llm_text(
        prompt=text,
        system_prompt=system_prompt,
        model="gpt-4.1-nano",
        user=username,
        temperature=0.3,
        max_tokens=4096,
    )

    return formatted_text
