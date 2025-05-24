#!/usr/bin/env python3
"""
Script to download sample media files using saved pyrogram messages.
Downloads audio, video, voice, and video_note files from sample messages.
"""

import asyncio
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

# Import from src directory (run script from project root)
from src.utils.download_media import download_file_3


async def download_sample_files():
    """Download sample files for audio, video, voice, and video_note media types."""
    
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    api_id = os.getenv("TELEGRAM_API_ID")
    if api_id is not None:
        api_id = int(api_id)
    api_hash = os.getenv("TELEGRAM_API_HASH")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    # Sample messages directory
    sample_messages_dir = Path(__file__).parent.parent / "sample_pyrogram_messages" / "sample_messages"
    
    # Output directory for downloaded files
    output_dir = Path(__file__).parent / "sample_data"
    output_dir.mkdir(exist_ok=True)
    
    # Media types to download
    media_types = ["audio", "video", "voice", "video_note"]
    
    for media_type in media_types:
        print(f"\n=== Downloading {media_type} sample ===")
        
        # Load the pickle file
        pickle_file = sample_messages_dir / f"sample_{media_type}_attached.pkl"
        
        if not pickle_file.exists():
            print(f"Warning: {pickle_file} not found, skipping...")
            continue
            
        try:
            # Load the message
            with open(pickle_file, "rb") as f:
                message = pickle.load(f)
            
            print(f"Loaded message ID: {message.id}")
            print(f"From user: {message.from_user.username}")
            
            # Download the file
            file_path = await download_file_3(
                message_id=message.id,
                username=message.from_user.username,
                target_dir=output_dir,
                api_id=api_id,
                api_hash=api_hash,
                bot_token=bot_token,
            )
            
            print(f"Successfully downloaded: {file_path}")
            
        except Exception as e:
            print(f"Error downloading {media_type}: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(download_sample_files())
