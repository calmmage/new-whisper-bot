#!/usr/bin/env python3
"""
Test script to check response formats from different LLM providers.

This script tests various LLM providers to see what cost/token information
they provide in their responses, which can be used for accurate cost tracking.

Usage:
    python scripts/check_response_formats.py
"""

import asyncio
import pickle
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import openai
from loguru import logger
from pydub import AudioSegment

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from botspot.components.new.llm_provider import aquery_llm_raw
from src.utils.cost_tracking import parse_cost


def inspect_response(response: Any, provider_name: str) -> Dict[str, Any]:
    """Inspect a response object and extract relevant information."""
    # Extract model name from provider name for parse_cost
    model_name = provider_name.split('_')[-1] if '_' in provider_name else "gpt-4.1-nano"
    
    info = {
        "provider": provider_name,
        "type": type(response).__name__,
        "has_usage": hasattr(response, 'usage'),
        "has_cost": False,
        "usage_info": None,
        "cost_info": None,
        "parsed_cost": None,
        "raw_response": str(response)[:500] + "..." if len(str(response)) > 500 else str(response)
    }
    
    # Try to parse cost using our utility
    try:
        parsed_cost = parse_cost(response, model_name)
        if parsed_cost is not None:
            info["parsed_cost"] = parsed_cost
            info["has_cost"] = True
    except Exception as e:
        info["parse_cost_error"] = str(e)
    
    # Check for usage information
    if hasattr(response, 'usage'):
        usage = response.usage
        info["usage_info"] = {
            "type": type(usage).__name__,
            "attributes": dir(usage),
            "data": {}
        }
        
        # Common usage attributes
        for attr in ['prompt_tokens', 'completion_tokens', 'total_tokens', 
                    'input_tokens', 'output_tokens', 'cache_read_input_tokens']:
            if hasattr(usage, attr):
                info["usage_info"]["data"][attr] = getattr(usage, attr)
    
    # Check for cost information
    cost_attrs = ['cost', 'price', 'billing', 'charges']
    for attr in cost_attrs:
        if hasattr(response, attr):
            info["has_cost"] = True
            if info["cost_info"] is None:
                info["cost_info"] = {}
            info["cost_info"][attr] = getattr(response, attr)
    
    # Check nested cost information
    if hasattr(response, 'usage') and response.usage:
        for attr in cost_attrs:
            if hasattr(response.usage, attr):
                info["has_cost"] = True
                if info["cost_info"] is None:
                    info["cost_info"] = {}
                info["cost_info"][f"usage.{attr}"] = getattr(response.usage, attr)
    
    return info


def create_test_audio() -> BytesIO:
    """Create a small test audio file for Whisper testing."""
    # Create a 5-second sine wave audio
    audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence
    
    # Export to BytesIO as MP3
    audio_buffer = BytesIO()
    audio.export(audio_buffer, format="mp3")
    audio_buffer.seek(0)
    audio_buffer.name = "test_audio.mp3"
    
    return audio_buffer


async def test_llm_providers() -> List[Dict[str, Any]]:
    """Test different LLM providers and collect response information."""
    results = []
    
    test_prompt = "Hello, this is a test message. Please respond with a short greeting."
    
    # Test different models
    models_to_test = [
        "claude-3.5-haiku",
        "claude-3.5-sonnet", 
        "claude-3.7",
        "o4-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4.1",
        "o3",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "grok-3-mini",
        "grok-3",
    ]
    
    for model in models_to_test:
        try:
            logger.info(f"Testing model: {model}")
            
            response = await aquery_llm_raw(
                prompt=test_prompt,
                user="petrlavrov",
                model=model,
                max_tokens=50
            )
            
            info = inspect_response(response, f"llm_provider_{model}")
            results.append(info)
            
            logger.info(f"✓ {model}: Usage={info['has_usage']}, Cost={info['has_cost']}")
            
        except Exception as e:
            logger.error(f"✗ Failed to test {model}: {e}")
            results.append({
                "provider": f"llm_provider_{model}",
                "error": str(e),
                "type": "error"
            })
    
    return results


async def test_openai_whisper() -> List[Dict[str, Any]]:
    """Test OpenAI Whisper API and collect response information."""
    results = []
    
    try:
        # Load config to get OpenAI API key
        import os
        
        client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create test audio
        test_audio = create_test_audio()
        
        logger.info("Testing OpenAI Whisper API")
        
        # Test transcription
        response = await client.audio.transcriptions.create(
            file=test_audio,
            model="whisper-1",
            response_format="text"
        )
        
        info = inspect_response(response, "openai_whisper_text")
        results.append(info)
        
        logger.info(f"✓ Whisper (text): Usage={info['has_usage']}, Cost={info['has_cost']}")
        
        # Test with verbose_json format to get more details
        test_audio.seek(0)
        response_verbose = await client.audio.transcriptions.create(
            file=test_audio,
            model="whisper-1",
            response_format="verbose_json"
        )
        
        info_verbose = inspect_response(response_verbose, "openai_whisper_verbose_json")
        results.append(info_verbose)
        
        logger.info(f"✓ Whisper (verbose_json): Usage={info_verbose['has_usage']}, Cost={info_verbose['has_cost']}")
        
    except Exception as e:
        logger.error(f"✗ Failed to test OpenAI Whisper: {e}")
        results.append({
            "provider": "openai_whisper",
            "error": str(e),
            "type": "error"
        })
    
    return results


async def test_openai_chat() -> List[Dict[str, Any]]:
    """Test OpenAI Chat API directly to compare with LLM provider."""
    results = []
    
    try:
        # Load config to get OpenAI API key
        import os
        
        client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        models_to_test = ["gpt-4.1-nano", "gpt-4o", "o4-mini"]
        
        for model in models_to_test:
            try:
                logger.info(f"Testing OpenAI direct: {model}")
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "Hello, this is a test message. Please respond with a short greeting."}
                    ],
                    max_tokens=50
                )
                
                info = inspect_response(response, f"openai_direct_{model}")
                results.append(info)
                
                logger.info(f"✓ OpenAI {model}: Usage={info['has_usage']}, Cost={info['has_cost']}")
                
            except Exception as e:
                logger.error(f"✗ Failed to test OpenAI {model}: {e}")
                results.append({
                    "provider": f"openai_direct_{model}",
                    "error": str(e),
                    "type": "error"
                })
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize OpenAI client: {e}")
        results.append({
            "provider": "openai_direct",
            "error": str(e),
            "type": "error"
        })
    
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of the results."""
    logger.info("\n" + "="*80)
    logger.info("RESPONSE FORMAT ANALYSIS SUMMARY")
    logger.info("="*80)
    
    providers_with_usage = []
    providers_with_cost = []
    providers_with_errors = []
    
    for result in results:
        if result.get("type") == "error":
            providers_with_errors.append(result["provider"])
            continue
            
        if result.get("has_usage"):
            providers_with_usage.append(result["provider"])
            
        if result.get("has_cost"):
            providers_with_cost.append(result["provider"])
    
    logger.info(f"\nProviders with usage information ({len(providers_with_usage)}):")
    for provider in providers_with_usage:
        logger.info(f"  ✓ {provider}")
    
    logger.info(f"\nProviders with cost information ({len(providers_with_cost)}):")
    for provider in providers_with_cost:
        logger.info(f"  💰 {provider}")
    
    logger.info(f"\nProviders with errors ({len(providers_with_errors)}):")
    for provider in providers_with_errors:
        logger.info(f"  ✗ {provider}")
    
    logger.info("\nDetailed usage information:")
    for result in results:
        if result.get("usage_info") and result["usage_info"]["data"]:
            logger.info(f"\n{result['provider']}:")
            for key, value in result["usage_info"]["data"].items():
                logger.info(f"  {key}: {value}")
            if result.get("parsed_cost") is not None:
                logger.info(f"  parsed_cost: ${result['parsed_cost']:.6f}")


async def main():
    """Main function."""
    from dotenv import load_dotenv
    load_dotenv()
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("scripts/logs/check_response_formats.log", level="DEBUG", rotation="10 MB")
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "response_formats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting response format analysis...")
    
    all_results = []
    
    # Test LLM providers
    logger.info("\n" + "-"*50)
    logger.info("Testing LLM Providers via botspot")
    logger.info("-"*50)
    llm_results = await test_llm_providers()
    all_results.extend(llm_results)
    
    # Test OpenAI Whisper
    logger.info("\n" + "-"*50)
    logger.info("Testing OpenAI Whisper API")
    logger.info("-"*50)
    whisper_results = await test_openai_whisper()
    all_results.extend(whisper_results)
    
    # Test OpenAI Chat directly
    logger.info("\n" + "-"*50)
    logger.info("Testing OpenAI Chat API directly")
    logger.info("-"*50)
    openai_results = await test_openai_chat()
    all_results.extend(openai_results)
    
    # Save results as pickle
    pickle_file = output_dir / "response_formats.pickle"
    with open(pickle_file, "wb") as f:
        pickle.dump(all_results, f)
    
    logger.info(f"\nResults saved to: {pickle_file}")
    
    # Print summary
    print_summary(all_results)
    
    # Save detailed results as text
    text_file = output_dir / "response_formats_detailed.txt"
    with open(text_file, "w") as f:
        for result in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"Provider: {result.get('provider', 'Unknown')}\n")
            f.write(f"Type: {result.get('type', 'Unknown')}\n")
            f.write(f"Has Usage: {result.get('has_usage', False)}\n")
            f.write(f"Has Cost: {result.get('has_cost', False)}\n")
            
            if result.get("error"):
                f.write(f"Error: {result['error']}\n")
            
            if result.get("usage_info"):
                f.write("\nUsage Info:\n")
                f.write(f"  Type: {result['usage_info']['type']}\n")
                f.write(f"  Data: {result['usage_info']['data']}\n")
            
            if result.get("cost_info"):
                f.write(f"\nCost Info: {result['cost_info']}\n")
            
            f.write(f"\nRaw Response (truncated):\n{result.get('raw_response', 'N/A')}\n")
    
    logger.info(f"Detailed results saved to: {text_file}")


if __name__ == "__main__":
    from botspot.core.bot_manager import BotManager
    bm = BotManager(
        admins_str="@petrlavrov"
        
    )
    asyncio.run(main())
