from textwrap import dedent

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from botspot import answer_safe, commands_menu, get_message_text, reply_safe
from botspot.components.new.llm_provider import aquery_llm_structured
from botspot.components.qol.bot_commands_menu import Visibility
from botspot.user_interactions import ask_user, ask_user_choice
from botspot.utils import markdown_to_html, send_safe
from botspot.utils.admin_filter import AdminFilter
from loguru import logger
from pydantic import BaseModel

from src.app import App

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
async def stats_handler(message: Message, app: App):
    """Stats command handler - shows usage statistics for all users"""
    assert message.from_user is not None

    # Get user statistics
    try:
        stats = await app.get_user_statistics()

        # Format the statistics nicely
        response = "<b>📊 Bot Usage Statistics</b>\n\n"

        # Summary section
        summary = stats["summary"]
        response += "<b>📈 Summary:</b>\n"
        response += f"• Total Users: {summary['total_users']}\n"
        response += f"• Total Requests: {summary['total_requests']}\n"
        response += f"• Total Audio Minutes: {summary['total_minutes']:.1f}\n"
        response += f"• Total Cost: ${summary['total_cost']:.4f}\n\n"

        # Per-user statistics
        response += "<b>👥 Per-User Statistics:</b>\n"

        user_stats = stats["user_stats"]
        # Sort users by total cost (descending)
        sorted_users = sorted(
            user_stats.items(), key=lambda x: x[1]["total_cost"], reverse=True
        )

        for username, user_data in sorted_users[:20]:  # Show top 20 users
            response += f"\n<b>@{username}</b>\n"
            response += f"  • Requests: {user_data['total_requests']}\n"
            response += f"  • Audio Minutes: {user_data['total_minutes']:.1f}\n"
            response += f"  • Total Cost: ${user_data['total_cost']:.4f}\n"

            # Show breakdown of operations if available
            if user_data["operations"]:
                response += "  • Operations: "
                op_details = []
                for op, op_data in user_data["operations"].items():
                    op_details.append(f"{op}({op_data['count']})")
                response += ", ".join(op_details) + "\n"

            # Show activity timeframe
            if user_data["first_activity"] and user_data["last_activity"]:
                first = user_data["first_activity"]
                last = user_data["last_activity"]
                response += f"  • Active: {first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')}\n"

        if len(user_stats) > 20:
            response += f"\n<i>... and {len(user_stats) - 20} more users</i>"

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        response = f"❌ Error retrieving statistics: {str(e)}"

    await send_safe(message.chat.id, response)


@commands_menu.botspot_command(
    "add_friend", "Add a friend to the bot (admin only)", visibility=Visibility.ADMIN_ONLY
)
@router.message(Command("add_friend"), AdminFilter())
async def add_friend_handler(message: Message, app: App, state: FSMContext):
    """Add a friend to the bot's friends list"""
    from botspot.user_interactions import get_username_from_command_or_dialog

    from src.utils.persistent_access_control import add_friend

    assert message.from_user is not None

    # Get username from command or via interactive dialog
    username = await get_username_from_command_or_dialog(
        message=message,
        state=state,
        cleanup=app.config.cleanup_messages,
        prompt="Please send the username or forward a message from the user you want to add as friend:",
        timeout=60,
    )

    if username is None:
        await send_safe(
            message.chat.id,
            "❌ Failed to get username. Operation cancelled.",
        )
        return

    # Add the friend
    try:
        success = await add_friend(username)
        if success:
            await send_safe(
                message.chat.id,
                f"✅ Successfully added {username} to friends list!",
            )
        else:
            await send_safe(
                message.chat.id,
                f"ℹ️ {username} is already in the friends list.",
            )
    except Exception as e:
        logger.error(f"Error adding friend {username}: {e}")
        await send_safe(
            message.chat.id,
            f"❌ Error adding friend: {str(e)}",
        )


@commands_menu.botspot_command(
    "remove_friend", "Remove a friend from the bot (admin only)", visibility=Visibility.ADMIN_ONLY
)
@router.message(Command("remove_friend"), AdminFilter())
async def remove_friend_handler(message: Message, app: App, state: FSMContext):
    """Remove a friend from the bot's friends list"""
    from botspot.user_interactions import get_username_from_command_or_dialog

    from src.utils.persistent_access_control import remove_friend

    assert message.from_user is not None

    # Get username from command or via interactive dialog
    username = await get_username_from_command_or_dialog(
        message=message,
        state=state,
        cleanup=app.config.cleanup_messages,
        prompt="Please send the username or forward a message from the user you want to remove from friends:",
        timeout=60,
    )

    if username is None:
        await send_safe(
            message.chat.id,
            "❌ Failed to get username. Operation cancelled.",
        )
        return

    # Remove the friend
    try:
        success = await remove_friend(username)
        if success:
            await send_safe(
                message.chat.id,
                f"✅ Successfully removed {username} from friends list!",
            )
        else:
            await send_safe(
                message.chat.id,
                f"ℹ️ {username} was not in the friends list.",
            )
    except Exception as e:
        logger.error(f"Error removing friend {username}: {e}")
        await send_safe(
            message.chat.id,
            f"❌ Error removing friend: {str(e)}",
        )


@commands_menu.botspot_command(
    "list_friends", "List all friends (admin only)", visibility=Visibility.ADMIN_ONLY
)
@router.message(Command("list_friends"), AdminFilter())
async def list_friends_handler(message: Message, app: App):
    """List all friends"""
    from src.utils.persistent_access_control import get_friends

    assert message.from_user is not None

    try:
        friends = await get_friends()

        if not friends:
            await send_safe(
                message.chat.id,
                "ℹ️ No friends in the list.",
            )
            return

        response = "<b>👥 Friends List:</b>\n\n"
        for i, friend in enumerate(friends, 1):
            response += f"{i}. {friend}\n"

        response += f"\n<i>Total: {len(friends)} friends</i>"

        await send_safe(message.chat.id, response)

    except Exception as e:
        logger.error(f"Error listing friends: {e}")
        await send_safe(
            message.chat.id,
            f"❌ Error listing friends: {str(e)}",
        )


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

    # Get and display cost information
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
