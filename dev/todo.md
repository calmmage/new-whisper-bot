# Todos

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

For future, NOT NOW
  - [ ] just 'reply to' activates gpt chat mode with that message text as context
  - [ ] add customized vocabulary / terms and send them to whisper api
