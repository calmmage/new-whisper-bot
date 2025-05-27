from aiogram import F, Router
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from botspot import answer_safe, commands_menu, get_message_text, reply_safe
from botspot.user_interactions import ask_user_choice
from botspot.utils import send_safe, markdown_to_html
from loguru import logger
from textwrap import dedent

from src.app import App

router = Router()


@commands_menu.botspot_command("start", "Start the bot")
@router.message(CommandStart())
async def start_handler(message: Message, app: App):
    assert message.from_user is not None
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
async def media_handler(message: Message, app: App, state: FSMContext):
    assert message.from_user is not None
    assert message.from_user.username is not None

    username = message.from_user.username

    # ask user
    model = await ask_user_choice(
        message.chat.id,
        "Please choose a model to use for transcription:",
        {
            "whisper-1": "Whisper 1 (Oldest, well tested, fast)",
            "gpt-4o-mini-transcribe": "GPT-4o Mini (Fast, new, not tested)",
            "gpt-4o-transcribe": "GPT-4o (Best, Slowest, most expensive)",
        },
        state=state,
        timeout=10,  # 10 seconds to choose, then default to whisper-1
        cleanup=True,
        default_choice=app.config.transcription_model,
    )

    # Send a processing message
    notif = await reply_safe(
        message, "Processing your media file. Estimating transcription time..."
    )

    # Transcribe the audio
    # Note: process_message already sets and clears per-user message_id internally
    transcription = await app.process_message(
        message,
        whisper_model=model,
        status_callback=create_notification_callback(notif),
    )
    await reply_safe(message, transcription)
    await notif.delete()

    if len(transcription) > app.config.summary_generation_threshold:
        # Create and send summary
        notif = await reply_safe(
            message, "Large transcript detected, creating summary..."
        )
        summary = await app.create_summary(
            transcription, username=username, message_id=message.message_id
        )

        # Clear message_id for this user
        app._user_message_ids.pop(username, None)

        await reply_safe(
            message,
            f"<b>AI Summary:</b>\n\n{summary}. \n\n <i>Reminder: you can reply to any message to chat about the transcript or summary with LLM.</i>",
        )
        await notif.delete()

    # Get and display cost information
    cost_info = await app.get_total_cost(username, message_id=message.message_id)
    total_cost = cost_info["total_cost"]
    if total_cost > 0.01:
        operation_costs = float(cost_info["operation_costs"])

        cost_breakdown = "\n".join(
            [
                f"  - {op.capitalize()}: ${cost:.4f}"
                for op, cost in operation_costs.items()
            ]
        )

        cost_message = (
            f"💰 <b>Processing Cost:</b>\n"
            f"Total: ${total_cost:.4f} USD\n"
            f"Breakdown:\n{cost_breakdown}"
        )

        await reply_safe(message, cost_message)


def create_notification_callback(notification_message: Message):
    text = notification_message.text
    assert text is not None

    async def callback(update_text: str):
        nonlocal text
        text = text + "\n" + update_text
        return await notification_message.edit_text(text)

    return callback


@router.message()
async def chat_handler(message: Message, app: App):
    if message.reply_to_message:
        await _reply_chat_handler(message, app)
    else:
        await _basic_chat_handler(message, app)


async def _reply_chat_handler(message: Message, app: App):
    """Use ai to allow user edit / chat about their transcript / summary"""
    assert message.reply_to_message is not None
    assert message.from_user is not None and message.from_user.username is not None

    username = message.from_user.username

    # Log start of chat operation
    logger.info(f"Processing chat request from user {username}")

    # todo: reconstruct full chat history from reply chain
    prompt = await get_message_text(message, include_reply=True)

    # Process chat request
    response = await app.chat_about_transcript(full_prompt=prompt, username=username, message_id=message.message_id)
    response = markdown_to_html(response)

    cost = await app.get_total_cost(username, message_id=message.message_id)
    chat_cost = cost["total_cost"]
    if chat_cost > 0.01:
        response += f"\n<i>cost: ${chat_cost:.4f} USD</i>"

    # Send response to user
    await reply_safe(
        message,
        response,
    )

    logger.info(f"Completed chat request for user {username}, cost: ${chat_cost:.4f}")


async def _basic_chat_handler(message: Message, app: App):
    """Basic chat handler"""
    await answer_safe(
        message,
        "This is a Whisper bot. "
        "\n\nSend an Audio, Video, Voice or Video Note message to transcribe and summarize it.",
        "\n\n<b>Reply</b> to a text message from bot to chat about its transcript or summary with LLM",
    )
