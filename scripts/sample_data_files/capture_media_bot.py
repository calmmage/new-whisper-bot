#!/usr/bin/env python3
"""
Simple Telegram bot to capture and save media files for testing.
Supports: audio, video, voice, video_note files.
"""

import asyncio
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from dotenv import load_dotenv
from loguru import logger

# Import our pyrogram download utility  
from src.utils.download_media import download_file_3


async def download_and_save_media(message: Message, media_type: str, bot: Bot):
    """Download media from message and save to sample_data directory using pyrogram."""
    
    # Create output directory
    output_dir = Path(__file__).parent / "sample_data"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Get API credentials from environment
        api_id = os.getenv("TELEGRAM_API_ID")
        if api_id is not None:
            api_id = int(api_id)
        api_hash = os.getenv("TELEGRAM_API_HASH")
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        
        # Use pyrogram to download (handles larger files than Bot API)
        file_path = await download_file_3(
            message_id=message.message_id,
            username=message.from_user.username,
            target_dir=output_dir,
            use_original_file_name=False,
            api_id=api_id,
            api_hash=api_hash,
            bot_token=bot_token,
        )
        
        logger.info(f"Downloaded {media_type}: {file_path}")
        await message.reply(f"✅ Saved {media_type} as: {file_path.name}")
        
    except Exception as e:
        logger.error(f"Error downloading {media_type}: {e}")
        await message.reply(f"❌ Error saving {media_type}: {e}")


async def main():
    """Main bot function."""
    
    # Load environment variables
    load_dotenv()
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")
    
    # Initialize bot and dispatcher
    bot = Bot(token=bot_token)
    dp = Dispatcher()
    
    # Handler for start command
    @dp.message(F.text.startswith("/start"))
    async def start_handler(message: Message):


        # import asyncio
        import pickle

        from dotenv import load_dotenv

        load_dotenv()

        # p = Path(__file__).parent.parent.parent / "scripts/sample_pyrogram_messages/sample_messages"
        p = Path("/Users/petrlavrov/work/experiments/new-whisper-bot/scripts/sample_pyrogram_messages/sample_messages")
        message = pickle.load(open(p / "sample_photo_attached.pkl", "rb"))
        # message = pickle.load(open(p / "sample_audio_attached.pkl", "rb"))

        api_id = os.getenv("TELEGRAM_API_ID")
        if api_id is not None:
            api_id = int(api_id)

        res = await download_file_3(
                message.id,
                message.from_user.username,
                api_id=api_id,
                api_hash=os.getenv("TELEGRAM_API_HASH"),
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            )
        print(res)

        await message.reply(
            "🎵 Media Capture Bot\n\n"
            "Send me:\n"
            "• Audio files 🎵\n"
            "• Video files 🎬\n" 
            "• Voice messages 🗣️\n"
            "• Video notes 📹\n\n"
            "I'll save them to the sample_data directory!"
            "\n\nDownloaded: {res}"
        )
    
    # Handler for audio files
    @dp.message(F.audio)
    async def audio_handler(message: Message):
        await download_and_save_media(message, "audio", bot)
    
    # Handler for video files
    @dp.message(F.video)
    async def video_handler(message: Message):
        await download_and_save_media(message, "video", bot)
    
    # Handler for voice messages
    @dp.message(F.voice)
    async def voice_handler(message: Message):
        await download_and_save_media(message, "voice", bot)
    
    # Handler for video notes
    @dp.message(F.video_note)
    async def video_note_handler(message: Message):
        await download_and_save_media(message, "video_note", bot)
    
    # Handler for text messages (excluding commands)
    @dp.message(F.text & ~F.text.startswith("/"))
    async def text_handler(message: Message):
        await message.reply(
            "Please send audio, video, voice, or video_note files.\n"
            "Use /start to see instructions."
        )
    
    logger.info("Starting media capture bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
