from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_price_per_1m: float  # Price per 1M input tokens
    output_price_per_1m: float  # Price per 1M output tokens
    currency: str = "USD"


# Model pricing database (prices per 1M tokens)
MODEL_PRICING: Dict[str, ModelPricing] = {
    # Anthropic Models
    "claude-3.5-haiku": ModelPricing(input_price_per_1m=0.80, output_price_per_1m=4.00),
    "claude-3-5-haiku": ModelPricing(input_price_per_1m=0.80, output_price_per_1m=4.00),
    "claude-3.5-sonnet": ModelPricing(
        input_price_per_1m=3.00, output_price_per_1m=15.00
    ),
    "claude-3-5-sonnet": ModelPricing(
        input_price_per_1m=3.00, output_price_per_1m=15.00
    ),
    "claude-3.7": ModelPricing(input_price_per_1m=3.00, output_price_per_1m=15.00),
    "claude-4": ModelPricing(input_price_per_1m=3.00, output_price_per_1m=15.00),
    "claude-4-sonnet": ModelPricing(input_price_per_1m=3.00, output_price_per_1m=15.00),
    # OpenAI Models - Cheap
    "gpt-4o-mini": ModelPricing(input_price_per_1m=0.15, output_price_per_1m=0.60),
    "o4-mini": ModelPricing(input_price_per_1m=0.55, output_price_per_1m=2.20),
    "gpt-4.1-mini": ModelPricing(input_price_per_1m=0.40, output_price_per_1m=1.60),
    "gpt-4.1-nano": ModelPricing(input_price_per_1m=0.10, output_price_per_1m=0.40),
    # OpenAI Models - Mid
    "gpt-4o": ModelPricing(input_price_per_1m=2.50, output_price_per_1m=10.00),
    "gpt-4.1": ModelPricing(input_price_per_1m=2.00, output_price_per_1m=8.00),
    # OpenAI Models - Max
    "o3": ModelPricing(input_price_per_1m=10.00, output_price_per_1m=40.00),
    "gpt-4": ModelPricing(input_price_per_1m=30.00, output_price_per_1m=60.00),
    "o1-pro": ModelPricing(input_price_per_1m=20.00, output_price_per_1m=80.00),
    # Google Models
    "gemini-2.5-flash": ModelPricing(input_price_per_1m=0.15, output_price_per_1m=0.60),
    "gemini-2.5-pro": ModelPricing(input_price_per_1m=1.25, output_price_per_1m=10.00),
    # xAI Models
    "grok-3-mini": ModelPricing(
        input_price_per_1m=1.00, output_price_per_1m=5.00
    ),  # Estimated
    "grok-3": ModelPricing(
        input_price_per_1m=5.00, output_price_per_1m=15.00
    ),  # Estimated
    # Legacy models for backward compatibility
    "gpt-3.5-turbo": ModelPricing(input_price_per_1m=1.50, output_price_per_1m=2.00),
    "claude-3-sonnet": ModelPricing(input_price_per_1m=8.00, output_price_per_1m=24.00),
    "claude-3-haiku": ModelPricing(input_price_per_1m=0.25, output_price_per_1m=1.25),
    # Whisper models (pricing per minute, stored as cost per minute * 1000 for consistency)
    "whisper-1": ModelPricing(
        input_price_per_1m=6.00, output_price_per_1m=0.0
    ),  # $0.006 per minute
}


def get_model_pricing_info(model: str) -> ModelPricing:
    """
    Get pricing information for a model.

    Args:
        model: Model name

    Returns:
        ModelPricing object with pricing info

    Raises:
        ValueError: If model pricing is not found
    """
    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try partial matches for model families
    for model_key, pricing in MODEL_PRICING.items():
        if model_key in model:
            return pricing

    # Default fallback to GPT-3.5 pricing
    return MODEL_PRICING["gpt-4o"]


def estimate_cost(
    input_tokens: float, 
    output_tokens: float, 
    model: str
) -> float:
    """
    Estimate cost based on token count and model.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name

    Returns:
        Estimated cost in USD
    """
    pricing = get_model_pricing_info(model)

    input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m

    return input_cost + output_cost


def estimate_cost_from_text(
    input_text: str, 
    output_text: str, 
    model: str,
    chars_per_token: float = 4.0
) -> float:
    """
    Estimate cost from text lengths using character-to-token heuristic.

    Args:
        input_text: Input text
        output_text: Output text
        model: Model name
        chars_per_token: Characters per token ratio (default: 4.0)

    Returns:
        Estimated cost in USD
    """
    input_tokens = len(input_text) / chars_per_token
    output_tokens = len(output_text) / chars_per_token

    return estimate_cost(input_tokens, output_tokens, model)


def estimate_whisper_cost(file_size_mb: float, model: str = "whisper-1", duration_seconds: float = None) -> float:
    """
    Estimate Whisper transcription cost based on audio duration or file size.

    Args:
        file_size_mb: File size in MB (used as fallback if duration_seconds is None)
        model: Whisper model name
        duration_seconds: Audio duration in seconds (if provided, used instead of file_size_mb)

    Returns:
        Estimated cost in USD
    """
    pricing = get_model_pricing_info(model)

    if duration_seconds is not None:
        # Convert seconds to minutes, rounding up to the nearest minute
        # Whisper charges per minute or part thereof
        estimated_minutes = (duration_seconds + 59) // 60
    else:
        # Fallback to rough estimate: 1MB ≈ 1 minute of audio
        estimated_minutes = file_size_mb

    # Whisper pricing is stored as $0.006 per minute, stored as 6.00 in input_price_per_1m
    return estimated_minutes * (pricing.input_price_per_1m / 1000)


def parse_cost(response: Any, model: str) -> Optional[float]:
    """
    Parse exact cost from LLM response if available.

    Args:
        response: LLM response object
        model: Model name

    Returns:
        Exact cost if available, None otherwise
    """
    # TODO: Implement parsing of actual usage data from different providers
    # This would extract token counts from response.usage if available

    if hasattr(response, 'usage'):
        usage = response.usage
        if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
            return estimate_cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                model=model
            )

    return None


def create_usage_info(
    input_length: int,
    output_length: int,
    processing_time: float,
    estimated_input_tokens: float,
    estimated_output_tokens: float,
    file_name: Optional[str] = None,
    file_size_mb: Optional[float] = None,
    audio_duration_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create standardized usage info dictionary.

    Args:
        input_length: Length of input text/data
        output_length: Length of output text/data
        processing_time: Time taken to process
        estimated_input_tokens: Estimated input tokens
        estimated_output_tokens: Estimated output tokens
        file_name: Optional file name
        file_size_mb: Optional file size in MB
        audio_duration_seconds: Optional audio duration in seconds

    Returns:
        Dictionary with usage information
    """
    usage_info = {
        "input_length": input_length,
        "output_length": output_length,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "processing_time": processing_time,
    }

    if file_name is not None:
        usage_info["file_name"] = file_name

    if file_size_mb is not None:
        usage_info["file_size_mb"] = file_size_mb

    # If audio duration is provided, use it for estimated minutes
    if audio_duration_seconds is not None:
        usage_info["audio_duration_seconds"] = audio_duration_seconds
        # Round up to nearest minute for Whisper pricing
        usage_info["estimated_minutes"] = (audio_duration_seconds + 59) // 60
    elif file_size_mb is not None:
        # Fallback to file size as rough estimate
        usage_info["estimated_minutes"] = file_size_mb

    return usage_info 
