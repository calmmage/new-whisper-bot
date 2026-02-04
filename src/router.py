from textwrap import dedent

from io import BytesIO

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, Message
from botspot import answer_safe, commands_menu, get_message_text, reply_safe
from botspot.components.new.llm_provider import aquery_llm_structured
from botspot.components.qol.bot_commands_menu import Visibility
from botspot.user_interactions import ask_user, ask_user_choice
from botspot.utils import markdown_to_html, send_safe
from botspot.utils.admin_filter import AdminFilter
from loguru import logger
from pydantic import BaseModel

from src.app import App
from src.bot_settings import get_bot_settings
from src.utils.stats_html import generate_stats_html

router = Router()


@commands_menu.botspot_command("start", "Start the bot")
@router.message(CommandStart())
async def start_handler(message: Message, app: App):
    assert message.from_user is not None
    welcome_message = dedent(
        f"""
        Hello! This is {app.name}!
        Send me an audio, video, voice, or document file to transcribe and summarize it.
        You can also <b>reply</b> to any text message from me to chat about its transcript or summary with LLM.
        """
    )
    await send_safe(message.chat.id, welcome_message)


@commands_menu.botspot_command("help", "Show this help message")
@router.message(Command("help"))
async def help_handler(message: Message, app: App):
    """Basic help command handler"""
    # todo: add language picker note
    help_message = dedent(
        f"""
        This is {app.name}!
        Send me an audio, video, voice, or document file to transcribe and summarize it.
        You can also <b>reply</b> to any text message from me to chat about its transcript or summary with LLM.
        
        You will be prompted to choose a model for transcription. There are three options:
        - <a href="https://platform.openai.com/docs/models/whisper-1">Whisper</a>: Oldest, well tested, fast
        - <a href="https://platform.openai.com/docs/models/gpt-4o-mini-transcribe">GPT-4o Mini</a>: Fast, new, sometimes confuses languages.
        - <a href="https://platform.openai.com/docs/models/gpt-4o-transcribe">GPT-4o</a>: Best, Slowest, most expensive
        
        Under the hood, the bot operates as follows:
        1. Download a media file
        2. Convert to mp3 and cut to smaller parts if necessary
        3. Load into memory and cut further into smaller chunks
        4. Tracnscribe using OpenAI api (whisper or GPT-4o-transcribe models)
        5. Format punctuation and capitalisation using gpt-4.1-nano.
        6. For larger transcripts, generate a summary using claude
        
        If the summary is bad, or you want to ask other questions using the transcript, you can reply to bot messages. The message you replied to (only that 1 message) will be added to context. Technically, this means that you can use this bot as a chatgpt (claude-4) proxy!
        
        You can find the source code <a href="https://github.com/calmmage/new-whisper-bot">here</a>
        Be aware that the bot uses my custom aiogram library <a href="https://github.com/calmmage/botspot">botspot</a> quite heavily
        So you might need to explore that as well to understand it. Basic template is available <a href="https://github.com/calmmage/botspot-template">here</a>
        """
    )
    await send_safe(message.chat.id, help_message)


@commands_menu.botspot_command(
    "stats", "Show usage statistics (admin only)", visibility=Visibility.ADMIN_ONLY
)
@router.message(Command("stats"), AdminFilter())
async def stats_handler(message: Message, app: App, state: FSMContext):
    """Stats command handler - shows usage statistics with menu options"""
    assert message.from_user is not None

    choice = await ask_user_choice(
        message.chat.id,
        "What statistics would you like to see?",
        {
            "all": "All users summary",
            "single": "Single user stats",
            "export": "Export HTML table",
        },
        state=state,
        timeout=30,
        default_choice="all",
    )

    if choice == "single":
        await _show_single_user_stats(message, app, state)
    elif choice == "export":
        await _export_stats_html(message, app)
    else:
        await _show_all_users_stats(message, app)


async def _show_all_users_stats(message: Message, app: App):
    """Show summary statistics for all users."""
    try:
        stats = await app.get_user_statistics()

        response = "<b>📊 Bot Usage Statistics</b>\n\n"

        summary = stats["summary"]
        response += "<b>📈 Summary:</b>\n"
        response += f"• Total Users: {summary['total_users']}\n"
        response += f"• Total Requests: {summary['total_requests']}\n"
        response += f"• Total Audio Minutes: {summary['total_minutes']:.1f}\n"
        response += f"• Total Cost: ${summary['total_cost']:.4f}\n\n"

        response += "<b>👥 Per-User Statistics:</b>\n"

        user_stats = stats["user_stats"]
        sorted_users = sorted(
            user_stats.items(), key=lambda x: x[1]["total_cost"], reverse=True
        )

        for username, user_data in sorted_users[:20]:
            response += f"\n<b>@{username}</b>\n"
            response += f"  • Requests: {user_data['total_requests']}\n"
            response += f"  • Audio Minutes: {user_data['total_minutes']:.1f}\n"
            response += f"  • Total Cost: ${user_data['total_cost']:.4f}\n"

            if user_data["operations"]:
                response += "  • Operations: "
                op_details = [
                    f"{op}({op_data['count']})"
                    for op, op_data in user_data["operations"].items()
                ]
                response += ", ".join(op_details) + "\n"

            if user_data["first_activity"] and user_data["last_activity"]:
                first = user_data["first_activity"]
                last = user_data["last_activity"]
                response += (
                    f"  • Active: {first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')}\n"
                )

        if len(user_stats) > 20:
            response += f"\n<i>... and {len(user_stats) - 20} more users</i>"

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        response = f"❌ Error retrieving statistics: {str(e)}"

    await send_safe(message.chat.id, response)


async def _show_single_user_stats(message: Message, app: App, state: FSMContext):
    """Show statistics for a single user."""
    username = await ask_user(
        message.chat.id,
        "Enter the username (with or without @):",
        state=state,
        timeout=60,
    )

    if not username:
        await send_safe(message.chat.id, "❌ No username provided.")
        return

    username = username.strip().lstrip("@")

    try:
        stats = await app.get_user_statistics()
        user_stats = stats["user_stats"]

        if username not in user_stats:
            await send_safe(message.chat.id, f"❌ No statistics found for @{username}")
            return

        user_data = user_stats[username]

        response = f"<b>📊 Statistics for @{username}</b>\n\n"
        response += f"• Total Requests: {user_data['total_requests']}\n"
        response += f"• Total Audio Minutes: {user_data['total_minutes']:.1f}\n"
        response += f"• Total Cost: ${user_data['total_cost']:.4f}\n\n"

        if user_data["operations"]:
            response += "<b>Operations breakdown:</b>\n"
            for op, op_data in user_data["operations"].items():
                response += f"  • {op}: {op_data['count']} times, ${op_data['cost']:.4f}\n"

        if user_data["models"]:
            response += "\n<b>Models used:</b>\n"
            for model, model_data in user_data["models"].items():
                response += f"  • {model}: {model_data['count']} times, ${model_data['cost']:.4f}\n"

        if user_data["first_activity"] and user_data["last_activity"]:
            first = user_data["first_activity"]
            last = user_data["last_activity"]
            response += f"\n<b>Activity period:</b>\n"
            response += f"  • First: {first.strftime('%Y-%m-%d %H:%M')}\n"
            response += f"  • Last: {last.strftime('%Y-%m-%d %H:%M')}\n"

    except Exception as e:
        logger.error(f"Error getting statistics for {username}: {e}")
        response = f"❌ Error retrieving statistics: {str(e)}"

    await send_safe(message.chat.id, response)


async def _export_stats_html(message: Message, app: App):
    """Export statistics as HTML table file."""
    try:
        stats = await app.get_user_statistics()
        html_content = generate_stats_html(stats)

        html_bytes = BytesIO(html_content.encode("utf-8"))
        document = BufferedInputFile(
            html_bytes.getvalue(), filename="bot_statistics.html"
        )

        await message.answer_document(
            document, caption="📊 Bot usage statistics export"
        )

    except Exception as e:
        logger.error(f"Error exporting statistics: {e}")
        await send_safe(message.chat.id, f"❌ Error exporting statistics: {str(e)}")


# Friend management commands are now handled by botspot's access_control component
# Enable with: BOTSPOT_ACCESS_CONTROL_ENABLED=true


@commands_menu.botspot_command(
    "settings", "Bot settings (admin only)", visibility=Visibility.ADMIN_ONLY
)
@router.message(Command("settings"), AdminFilter())
async def settings_handler(message: Message, state: FSMContext):
    """Admin settings menu."""
    settings = get_bot_settings()

    choice = await ask_user_choice(
        message.chat.id,
        "Select a setting to toggle:",
        {
            "show_cost": "Show cost info after processing",
            "back": "Back (no changes)",
        },
        state=state,
        timeout=30,
        default_choice="back",
    )

    if choice == "show_cost":
        new_value = await settings.toggle("show_cost_info")
        status = "enabled" if new_value else "disabled"
        await send_safe(message.chat.id, f"✅ Cost info display is now <b>{status}</b>")
    else:
        await send_safe(message.chat.id, "No changes made.")


class Language(BaseModel):
    language_code: str


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
        cleanup=app.config.cleanup_messages,
        default_choice=app.config.transcription_model,
    )

    language_code = await ask_user_language(message, app, state)

    # Note: speedup will be asked later if file is processed on disk

    # Send a processing message
    notif = await reply_safe(
        message, "Processing your media file. Estimating transcription time..."
    )

    # Transcribe the audio
    transcription = await app.process_message(
        message,
        state,
        whisper_model=model,
        language=language_code,
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
        summary = markdown_to_html(summary)

        await reply_safe(
            message,
            f"<b>AI Summary:</b>\n\n{summary}. \n\n <i>Reminder: you can reply to any message to chat about the transcript or summary with LLM.</i>",
        )
        await notif.delete()

    # Get and display cost information (if enabled in settings)
    settings = get_bot_settings()
    show_cost = await settings.get("show_cost_info")
    if show_cost:
        cost_info = await app.get_total_cost(username, message_id=message.message_id)
        total_cost = float(cost_info["total_cost"])
        if total_cost > 0.01:
            operation_costs = cost_info["operation_costs"]

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

    async def callback(update_text: str):
        nonlocal text
        assert text is not None
        text = text + "\n" + update_text
        await notification_message.edit_text(text)

    return callback


async def ask_user_language(message: Message, app: App, state: FSMContext):
    assert message.from_user is not None
    username = message.from_user.username
    language = await ask_user_choice(
        message.chat.id,
        "Please choose a language for transcription:",
        {
            "auto": "Auto-detect",
            "en": "English",
            "ru": "Russian",
            "other": "Other (enter manually)",
        },
        state=state,
        default_choice="auto",
        timeout=10,
        cleanup=app.config.cleanup_messages,
    )
    if language == "auto":
        language_code = None
    elif language == "other":
        language_str = await ask_user(
            message.chat.id,
            "Please enter the language manually:",
            state=state,
            cleanup=app.config.cleanup_messages,
        )
        # use simple llm query to
        if language_str is None:
            await message.reply("No language provided, please retry.")
            return
        parsed_language: Language = await aquery_llm_structured(
            prompt=language_str,
            output_schema=Language,
            model="gpt-4.1-nano",
            system_message="You are a language detection assistant. Your goal is to return the language code in ISO 639-1 format (e.g., 'en' for English, 'ru' for Russian).",
            user=username,
        )
        language_code = parsed_language.language_code
        logger.info(
            f"User input: {language_str}, detected language code: {language_code}"
        )
    else:
        language_code = language
    return language_code


async def ask_user_speedup(message: Message, app: App, state: FSMContext):
    speedup = await ask_user_choice(
        message.chat.id,
        "Please choose audio speedup:",
        {
            "none": "No speedup (original speed)",
            "2": "2x speed",
            "3": "3x speed",
            "4": "4x speed",
            "5": "5x speed",
        },
        state=state,
        default_choice="2",
        timeout=10,
        cleanup=app.config.cleanup_messages,
    )

    if speedup == "none":
        return None
    else:
        return float(speedup)


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
    response = await app.chat_about_transcript(
        full_prompt=prompt, username=username, message_id=message.message_id
    )
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
        "\n\nSend an Audio, Video, Voice or Video Note message to transcribe and summarize it."
        "\n\n<b>Reply</b> to a text message from bot to chat about its transcript or summary with LLM",
    )
