from aiogram import F, Router, html
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from botspot import answer_safe, commands_menu, get_message_text, reply_safe
from botspot.components.new.llm_provider import aquery_llm_text
from botspot.user_interactions import ask_user_choice
from botspot.utils import send_safe

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

    # ask user
    model = await ask_user_choice(
        message.chat.id,
        "Please choose a model to use for transcription:",
        {
            "whisper-1": "Whisper 1 (Oldest, tested)",
            "gpt-4o-mini-transcribe": "GPT-4o Mini (?)",
            "gpt-4o-transcribe": "GPT-4o (Best, Slowest, most expensive)",
        },
        state=state,
        timeout=10,  # 10 seconds to choose, then default to whisper-1
        cleanup=True,
        default_choice="whisper-1",
    )

    # Send a processing message
    notif = await reply_safe(
        message, "🔄 Processing your media file... This may take a few minutes."
    )

    # Transcribe the audio
    transcription = await app.run(message.message_id, username, model=model)
    await reply_safe(message, transcription)
    await notif.delete()

    # todo: save info - requests, results. usage stats - somewhere (to mongo?).
    # Create and send summary
    notif = await reply_safe(message, "🔄 Creating summary...")
    summary = await app.create_summary(transcription, username=username)
    await reply_safe(message, f"📋 <b>Summary:</b>\n\n{summary}")

    # todo: tell user how much this costed me.
    await notif.delete()


@router.message()
async def chat_handler(message: Message, app: App):
    if message.reply_to_message:
        await _reply_chat_handler(message, app)
    else:
        await _basic_chat_handler(message, app)


async def _reply_chat_handler(message: Message, app: App):
    """Use ai to allow user edit / chat about their transcript / summary"""
    assert message.reply_to_message is not None

    # todo: reconstruct full chat history from reply chain
    prompt = get_message_text(message, include_reply=True)
    response = await aquery_llm_text(prompt=prompt, model=app.config.summary_model)
    await reply_safe(
        message,
        response,
        # parse_mode="HTML",
    )


async def _basic_chat_handler(message: Message, app: App):
    """Basic chat handler"""
    await answer_safe(
        message,
        "This is a Whisper bot. "
        "\n\nSend an Audio, Video, Voice or Video Note message to transcribe and summarize it.",
    )
