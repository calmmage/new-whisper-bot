import threading
import time
from pathlib import Path

import psutil
from aiogram import F, Router, html
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from botspot import commands_menu, reply_safe
from botspot.utils import send_safe
from botspot.utils.unsorted import get_message_attachments
from loguru import logger

from app import App
from src.app import App

router = Router()


@commands_menu.botspot_command("start", "Start the bot")
@router.message(CommandStart())
async def start_handler(message: Message, app: App):
    await send_safe(
        message.chat.id,
        f"Hello, {html.bold(message.from_user.full_name)}!\n"
        f"Welcome to {app.name}!\n"
        f"Use /help to see available commands.",
    )


# @commands_menu.botspot_command("help", "Show this help message")
# @router.message(Command("help"))
# async def help_handler(message: Message, app: App):
#     """Basic help command handler"""
#     await send_safe(
#         message.chat.id,
#         f"This is {app.name}. Use /start to begin."
#         "Available commands:\n"
#         "/start - Start the bot\n"
#         "/help - Show this help message\n"
#         "/help_botspot - Show Botspot help\n"
#         "/timezone - Set your timezone\n"
#         "/error_test - Test error handling",
#     )


@router.message(F.audio | F.voice | F.video | F.document | F.video_note)
async def main_chat_handler(message: Message, app: App, state: FSMContext):
    assert message.from_user is not None
    assert message.from_user.username is not None

    username = message.from_user.username

    # Send a processing message
    notif = await reply_safe(message, "🔄 Processing your media file... This may take a few minutes.")

    # Transcribe the audio
    transcription = await app.run(message.message_id, username)
    await reply_safe(message, transcription)
    await notif.delete()

    # Create and send summary
    notif = await reply_safe(message, "🔄 Creating summary...")
    summary = await app.create_summary(transcription, username=username)
    await reply_safe(message, f"📋 <b>Summary:</b>\n\n{summary}")
    await notif.delete()
