from aiogram import F, Router, html
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from botspot import answer_safe, commands_menu, get_message_text, reply_safe
from botspot.user_interactions import ask_user_choice
from botspot.utils import send_safe
from loguru import logger

# from src.old.app import App
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
    # todo: make a nicer message - time estimate - here, and update it along the way.
    notif = await reply_safe(
        message, "🔄 Processing your media file... This may take a few minutes."
    )

    # Transcribe the audio
    # Note: process_message already sets and clears _current_message_id internally
    transcription = await app.process_message(message, whisper_model=model)
    await reply_safe(message, transcription)
    await notif.delete()

    # Create and send summary
    notif = await reply_safe(message, "🔄 Creating summary...")

    # Set message_id context variable
    app._current_message_id = message.message_id

    summary = await app.create_summary(transcription, username=username)

    # Clear message_id context variable
    app._current_message_id = None

    await reply_safe(message, f"📋 <b>Summary:</b>\n\n{summary}")
    await notif.delete()

    # Get and display cost information
    cost_info = await app.get_total_cost(username)
    total_cost = cost_info["total_cost"]
    operation_costs = cost_info["operation_costs"]

    cost_breakdown = "\n".join([
        f"  - {op.capitalize()}: ${cost:.4f}" 
        for op, cost in operation_costs.items()
    ])

    cost_message = (
        f"💰 <b>Processing Cost:</b>\n"
        f"Total: ${total_cost:.4f} USD\n"
        f"Breakdown:\n{cost_breakdown}"
    )

    await reply_safe(message, cost_message)


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

    # Get cost before chat operation
    before_cost = await app.get_total_cost(username)

    # Set message_id context variable
    app._current_message_id = message.message_id

    # Process chat request
    response = await app.chat_about_transcript(
        full_prompt=prompt, username=username
    )

    # Clear message_id context variable
    app._current_message_id = None

    # Send response to user
    await reply_safe(
        message,
        response,
        # parse_mode="HTML",
    )

    # Get cost after chat operation
    after_cost = await app.get_total_cost(username)

    # Calculate cost of this operation
    chat_cost = after_cost["total_cost"] - before_cost["total_cost"]

    # Only show cost if it's significant
    if chat_cost > 0.0001:
        cost_message = f"💬 <b>Chat Cost:</b> ${chat_cost:.4f} USD"
        await reply_safe(message, cost_message)

    logger.info(f"Completed chat request for user {username}, cost: ${chat_cost:.4f}")


async def _basic_chat_handler(message: Message, app: App):
    """Basic chat handler"""
    await answer_safe(
        message,
        "This is a Whisper bot. "
        "\n\nSend an Audio, Video, Voice or Video Note message to transcribe and summarize it.",
    )
