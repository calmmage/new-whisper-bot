from pathlib import Path
from typing import Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings



class AppConfig(BaseSettings):
    """Basic app configuration"""

    telegram_bot_token: SecretStr
    telegram_api_id: int
    telegram_api_hash: SecretStr

    # todo: use
    transcription_model: str = "whisper-1"  # Default transcription model
    summary_model: str = "claude-4-sonnet"
    # todo: use
    chat_model: str = (
        "claude-4-sonnet"  # Default chat model for discussing transcripts and summaries
    )
    # todo: use
    formatting_model: str = "gpt-4.1-nano"

    downloads_dir: Path = Path("downloads").absolute()
    cleanup_downloads: bool = True

    openai_api_key: Optional[SecretStr] = None
    use_memory_profiler: bool = False

    # todo: delete unused fields
    # Whisper API and audio chunking settings
    # whisper_chunk_duration: int = 600  # 10 minutes in seconds
    whisper_chunk_duration: int = 120  # 2 minutes in seconds
    # whisper_overlap_duration: int = 30  # 30 seconds
    whisper_overlap_duration: int = 5  # 30 seconds
    # actually, my account says 7500 rpm
    # todo: disentangle the cutting logic from rate limit, set target chunk amount instead.
    whisper_rate_limit: int = 50  # Maximum number of chunks to create
    # not sure - maybe 100? This is a global limit so need to be careful to not shoot me in the leg.
    # todo: double-check
    whisper_max_concurrent: int = 50  # Maximum number of concurrent API calls
    # todo: file size limit is 25 megabytes - look out for that instead..

    target_chunk_count: int = 20
    max_chunk_size: int = 20 * 60  # 20 min
    min_chunk_size: int = 2 * 60  # 2 min
