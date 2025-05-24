import pickle
from pathlib import Path

import pytest

# from aiogram.types import Message
from pyrogram.types import Message as PyrogramMessage

# from dev.pyrogram_file_loader import extract_file_name_from_pyrogram_message

SAMPLE_DIR = Path(__file__).parent / "sample_messages"


def extract_file_name_from_pyrogram_message(
    msg: PyrogramMessage, use_original_file_name: bool = True
) -> str:
    # Try document, audio, video (these have file_name) if using original names
    if use_original_file_name:
        for attr in ("document", "audio", "video"):
            media = getattr(msg, attr, None)
            if media and getattr(media, "file_name", None):
                return media.file_name

    # Generate ID-based names or fallback for media without original names
    if msg.document and not use_original_file_name:
        # For documents, try to preserve extension from original filename
        original_name = getattr(msg.document, "file_name", None)
        if original_name and "." in original_name:
            ext = original_name.split(".")[-1]
            short_id = msg.document.file_id[-12:]
            return f"doc_{short_id}.{ext}"
        else:
            short_id = msg.document.file_id[-12:]
            return f"doc_{short_id}.bin"

    if msg.audio and not use_original_file_name:
        # For audio, try to preserve extension from original filename
        original_name = getattr(msg.audio, "file_name", None)
        if original_name and "." in original_name:
            ext = original_name.split(".")[-1]
            short_id = msg.audio.file_id[-12:]
            return f"audio_{short_id}.{ext}"
        else:
            short_id = msg.audio.file_id[-12:]
            return f"audio_{short_id}.mp3"  # Default audio extension

    if msg.video and not use_original_file_name:
        # For video, try to preserve extension from original filename
        original_name = getattr(msg.video, "file_name", None)
        if original_name and "." in original_name:
            ext = original_name.split(".")[-1]
            short_id = msg.video.file_id[-12:]
            return f"video_{short_id}.{ext}"
        else:
            short_id = msg.video.file_id[-12:]
            return f"video_{short_id}.mp4"  # Default video extension

    # Voice, video_note, photo: always generate ID-based names (no original names available)
    if msg.voice:
        short_id = msg.voice.file_id[-12:]
        return f"voice_{short_id}.ogg"
    if msg.video_note:
        short_id = msg.video_note.file_id[-12:]
        return f"video_note_{short_id}.mp4"
    if msg.photo:
        short_id = msg.photo.file_id[-12:]
        # For photos, we don't know the format, so use a generic image extension
        # Could be jpg, png, webp, etc. - default to jpg as most common
        return f"photo_{short_id}.jpg"
    # Fallback
    return "file.bin"


@pytest.mark.parametrize(
    "pkl_file, use_original, expected_file_name",
    [
        # Test with original file names (default behavior)
        (
            SAMPLE_DIR / "sample_audio_attached.pkl",
            True,
            "GMT20250522-065206_Recording.m4a",
        ),
        (SAMPLE_DIR / "sample_video_attached.pkl", True, "IMG_3025.MOV"),
        (SAMPLE_DIR / "sample_document_attached.pkl", True, "receipt_17.05.2025.pdf"),
        # Voice, video_note, photo always use ID-based names regardless of flag
        (SAMPLE_DIR / "sample_voice_attached.pkl", True, "voice_oexmGcmvTHgQ.ogg"),
        (
            SAMPLE_DIR / "sample_video_note_attached.pkl",
            True,
            "video_note_sF4Dv8MUaHgQ.mp4",
        ),
        (SAMPLE_DIR / "sample_photo_attached.pkl", True, "photo_MCAAN5AAceBA.jpg"),
        # Test with shortened ID-based names
        (
            SAMPLE_DIR / "sample_audio_attached.pkl",
            False,
            "audio_ymGcmvTHgQ.m4a",
        ),  # Will need to check actual ID
        (
            SAMPLE_DIR / "sample_video_attached.pkl",
            False,
            "video_ymGcmvTHgQ.MOV",
        ),  # Will need to check actual ID
        (
            SAMPLE_DIR / "sample_document_attached.pkl",
            False,
            "doc_ymGcmvTHgQ.pdf",
        ),  # Will need to check actual ID
        # Voice, video_note, photo same as above since they don't have original names
        (SAMPLE_DIR / "sample_voice_attached.pkl", False, "voice_oexmGcmvTHgQ.ogg"),
        (
            SAMPLE_DIR / "sample_video_note_attached.pkl",
            False,
            "video_note_sF4Dv8MUaHgQ.mp4",
        ),
        (SAMPLE_DIR / "sample_photo_attached.pkl", False, "photo_MCAAN5AAceBA.jpg"),
    ],
)
def test_extract_file_name(pkl_file, use_original, expected_file_name):
    msg = pickle.load(open(pkl_file, "rb"))
    file_name = extract_file_name_from_pyrogram_message(
        msg, use_original_file_name=use_original
    )
    print(f"{pkl_file.name} (use_original={use_original}): {file_name}")
    assert file_name == expected_file_name
    assert isinstance(file_name, str) and len(file_name) > 0
