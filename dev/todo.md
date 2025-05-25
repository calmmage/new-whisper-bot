# Todos

## 25 May 2025

- [x] cleanup original file - if not mp3
- [x] cleanup mp3 file
- [x] cleanup chunks after parsing
- [ ] use "count ... " - which ... calculates how many chunks to cut thing into
  - [ ] process all chunks in parallel, not one by one
- [x] set timeout on ask_user - just like 10 seconds
- [x] fix 'summary' command - model or stuff.

- [ ] debug and fix merge chunks - what did break? Add tests? 

- [ ] rework audio cutting logic - it is very convoluted now... 
  - if small -> use old cutting (not implemented yet) 
  - if large - cut into chunks. Avoid too many chunks to speed-up stuff.. 

- [ ] better formatting? html, md or something.. 
  - Send as pdf or as Notion doc? 
- [ ] track usage - incoming requests and generated transcriptions
- [ ] report to a user monetary cost of what he just parsed
- [ ] add a command to show total usage stats for current user or for admin. 
- [ ] add a basic chat handler explaining to a user how to use the bot
  - [ ] add a gpt chat handler to chat with gpt use transcript as ... reply message 
  - [ ] add warmup messages to all llm_provider handlers
- [ ] add a better estimate for how long it will take to process the audio
  - [ ] report stages of processing  to a user

- [ ] for small files (below ? 20 minutes?) use old in-memory pydub cut audio.. 

Bugfix
- [ ] Merge chunks - seems broken now

Finish
- [ ] apply basic formatting to text chunks.

Rework
- [ ] rework audio cutting: use ffmpeg to cut in-place first into big chunks, then pydub to cut in-memory









## Plan

- a - Download audio
- b - Cut audio in-place (profile memory?)
- c - Parse audio with whisper
- d - use claude to generate summary / bullet-points
- e - extract audio from video
- f - convert to mp3 in-place?

features
- 1) Parse audio into ...
- 2) Bullet-points summary
- 3) Usage limits? Friends and stuff?

Old features / code that I will need
- cut audio with overlap
- merge text with overlap

## Key components
- [x] download
- [x] convert
  - [x] add two separate implementations - with and without memory profiler. and a wrapper with a flag to select one
- [x] cut
  - [x] add two separate implementations - with and without memory profiler. and a wrapper with a flag to select one
- [x] parse chunks
  - [x] use OpenAI Whisper API with retry logic
  - [x] add rate limiting with backoff retry
- [x] merge & format
  - [x] implemented with custom text_utils for overlap detection
  - [x] find the logic in old wihsper bot and use that. difflib wouldn't do.
  - [x] add nice paragraph formatting and punctuation as in old bot
- [x] summarize
  - [x] implemented with botspot llm provider
  - [x] use tested system prompt
  - [ ] add an option for user to do additional freeform text requests in chat (1 follow-up?)

## Done











# Old todos

## Human checks
- [x] download
- [x] convert to mp3
- [x] cut audio
- [ ] transcribe
- [ ] format and merge
- [ ] summarize

For future, NOT NOW
  - [x] just 'reply to' activates gpt chat mode with that message text as context
  - [ ] add customized vocabulary / terms and send them to whisper api  
    - [ ] "есть ли какие-то уникальные словечки которые тебе важно чтобы whisper распознал правильно"?


- [x] add 3 models - 'whisper-1', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe'
  - [ ] compare prices (using check-script? transcribe a small audio first to see model response format and if it has prices / tokens)
  - [x] allow user to pick which model to use for transcription (use ask_user_choice), default - whisper-1
- [x] cleanup - delete all final and intermediary media files
  - [x] original video / audio file
  - [x] chunks
  - [x] converted to mp3

# Done