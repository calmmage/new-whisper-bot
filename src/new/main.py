def process_audio():
    in_memory_audio = download_audio_to_memory(message)

    audio_file_size = calculate_audio_size(in_memory_audio)
    AUDIO_SIZE_THRESHOLD = 100 * 1024 * 1024  # 50 MB
    # offload files over 50 megabytes to disk
