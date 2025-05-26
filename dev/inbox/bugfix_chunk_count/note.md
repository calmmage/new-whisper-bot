02:47:11 | INFO | src.core.convert_to_mp3:93 | Successfully converted /app/downloads/GMT20250522-065206_Recording.m4a to
/app/downloads/GMT20250522-065206_Recording.mp3
02:47:11 | INFO | src.core.cut_audio:85 | Audio duration: 6114.672 seconds
02:47:11 | INFO | src.core.cut_audio:120 | Cutting audio into 1 chunks of 120000 seconds with 5 seconds overlap
02:47:11 | INFO | src.core.cut_audio:172 | Cutting audio into pieces (standard implementation)
02:47:11 | INFO | src.core.cut_audio:190 | Creating 1 chunks with 5s overlap
02:48:01 | INFO | src.core.cut_audio:238 | Created chunk 1/1:
/app/downloads/GMT20250522-065206_Recording_chunks/GMT20250522-065206_Recording_chunk_000.mp3
02:48:01 | INFO | src.core.parse_audio_chunks:98 | Transcribing 1 audio chunks
Failed to fetch updates - TelegramNetworkError: HTTP Client says - Request timeout error
Sleep for 1.000000 seconds and try again... (tryings = 0, bot id = 6319751885)
02:48:02 | INFO | src.core.parse_audio_chunks:48 | Transcribing
/app/downloads/GMT20250522-065206_Recording_chunks/GMT20250522-065206_Recording_chunk_000.mp3 using OpenAI Whisper API
02:48:15 | ERROR | src.core.parse_audio_chunks:74 | Error transcribing audio chunk
/app/downloads/GMT20250522-065206_Recording_chunks/GMT20250522-065206_Recording_chunk_000.mp3: Error code: 413 -
{'error': {'message': '413: Maximum content size limit (26214400) exceeded (26224592 bytes read)', 'type': '
server_error', 'param': None, 'code': None}}
02:48:15 | ERROR | botspot.components.middlewares.error_handler:44 |
File "/app/src/router.py", line 70, in main_chat_handler
transcription = await app.run(message.message_id, username, model=model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/app/src/app.py", line 104, in run
transcription_pieces = await self.parse_audio_chunks(audio_pieces, model=model)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/app/src/app.py", line 166, in parse_audio_chunks
transcribed_chunks = await parse_audio_chunks(
^^^^^^^^^^^^^^^^^^^^^^^^^
File "/app/src/core/parse_audio_chunks.py", line 113, in parse_audio_chunks
transcriptions = await asyncio.gather(*tasks)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/app/src/core/parse_audio_chunks.py", line 107, in process_chunk
return await parse_audio_chunk(chunk_path, model_name, language, api_key)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/tenacity/asyncio/__init__.py", line 189, in async_wrapped
return await copy(fn, *args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/tenacity/asyncio/__init__.py", line 111, in __call__
do = await self.iter(retry_state=retry_state)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/tenacity/asyncio/__init__.py", line 153, in iter
result = await action(retry_state)
^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/tenacity/_utils.py", line 99, in inner
return call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/tenacity/__init__.py", line 400, in <lambda>
self._add_action_func(lambda rs: rs.outcome.result())
^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/concurrent/futures/_base.py", line 449, in result
return self.__get_result()
^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
raise self._exception
File "/usr/local/lib/python3.12/site-packages/tenacity/asyncio/__init__.py", line 114, in __call__
result = await fn(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^
File "/app/src/core/parse_audio_chunks.py", line 61, in parse_audio_chunk
response = await client.audio.transcriptions.create(
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/openai/resources/audio/transcriptions.py", line 712, in create
return await self._post(
^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/openai/_base_client.py", line 1742, in post
return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/site-packages/openai/_base_client.py", line 1549, in request
raise self._make_status_error_from_response(err.response) from None
openai.APIStatusError: Error code: 413 - {'error': {'message': '413: Maximum content size limit (26214400) exceeded (
26224592 bytes read)', 'type': 'server_error', 'param': None, 'code': None}}
