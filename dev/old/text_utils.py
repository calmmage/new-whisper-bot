from difflib import SequenceMatcher
from typing import Iterable

import loguru
from bot_base.utils.gpt_utils import (
    arun_command_with_gpt,
    get_token_count,
    token_limit_by_model,
)


def normalize_text(text):
    return text.lower().replace(".", "").replace(",", "")


def find_overlap(text1, text2):
    s = SequenceMatcher(None, text1, text2)
    matching_blocks = s.get_matching_blocks()

    res_match = matching_blocks[0]
    for match in matching_blocks:
        if match.size > res_match.size:
            res_match = match

    # max_overlap_text = text1[match.a:match.a + match.size]
    return res_match


def find_segment_pos(segment: str, text: str):
    segment = segment.strip().lower()
    text = text.lower()

    beg = 0
    i = 0
    j = 0
    while i < len(segment):
        if i == 0:
            beg = j
        if not segment[i].isalnum():
            i += 1
            continue
        if not text[j].isalnum():
            j += 1
            continue

        if segment[i] == text[j]:
            i += 1
            j += 1
        else:
            if i > 0:
                i = 0
                j = beg + 1
            else:
                j += 1
        if j >= len(text):
            break

    return beg, j


DEFAULT_BUFFER = 25
DEFAULT_MATCH_CUTOFF = 15


def merge_two_chunks(
    chunk1,
    chunk2,
    buffer=DEFAULT_BUFFER,
    match_cutoff=DEFAULT_MATCH_CUTOFF,
    logger=None,
):
    if logger is None:
        logger = loguru.logger
    ending = " ".join(chunk1.split()[-buffer:])
    beginning = " ".join(chunk2.split()[:buffer])
    N = len(ending)
    M = len(beginning)
    ending = normalize_text(ending)
    beginning = normalize_text(beginning)

    # find maximum overlap
    match = find_overlap(ending, beginning)

    logger.debug(f"{ending=}")
    logger.debug(f"{beginning=}")
    logger.debug(f"Overlap size: {match.size}")
    segment = ending[match.a : match.a + match.size].strip()
    if match.size > 1:
        logger.debug(f"Overlap text: {segment}")
    if match.size < match_cutoff:
        logger.warning("Overlap is too small, merging as is")
        return chunk1 + chunk2

    pos1 = find_segment_pos(segment, chunk1[-N:].lower())
    pos1 = (pos1[0] + len(chunk1) - N, pos1[1] + len(chunk1) - N)
    pos2 = find_segment_pos(segment, chunk2[:M].lower())

    logger.debug(f"text in ending: {chunk1[pos1[0] : pos1[1]]}")
    logger.debug(f"text in beginning: {chunk2[pos2[0] : pos2[1]]}")
    return chunk1[: pos1[1]] + chunk2[pos2[1] :]


def merge_all_chunks(
    chunks: Iterable,
    buffer=DEFAULT_BUFFER,
    match_cutoff=DEFAULT_MATCH_CUTOFF,
    logger=None,
):
    """merge chunks using reduce method"""
    result = ""
    for chunk in chunks:
        result = merge_two_chunks(
            result, chunk, buffer=buffer, match_cutoff=match_cutoff, logger=logger
        )
    return result


FORMAT_TEXT_COMMAND = """
You're text formatting assistant. Your goal is:
- Add rich punctuation - new lines, quotes, dots and commas where appropriate
- Break the text into paragraphs using double new lines
Output language: Same as input
"""
FIX_GRAMMAR_COMMAND = """
- Fix grammar and typos
"""
KEEP_GRAMMAR_COMMAND = """
- keep the original text word-to-word, with only minor changes where absolutely necessary
"""


async def format_text_with_gpt(
    text, model="gpt-3.5-turbo", fix_grammar_and_typos=False, logger=None
):
    if logger is None:
        logger = loguru.logger
    token_limit = token_limit_by_model[model]
    token_count = get_token_count(text, model=model)
    logger.debug(f"{token_count=}, {token_limit=}")

    if token_count > token_limit / 2:
        raise ValueError(f"Text is too long: {token_count} > {token_limit / 2}")
    if fix_grammar_and_typos:
        command = FORMAT_TEXT_COMMAND + FIX_GRAMMAR_COMMAND
    else:
        command = FORMAT_TEXT_COMMAND + KEEP_GRAMMAR_COMMAND
    return await arun_command_with_gpt(command, text, model=model)


import re

MAX_TELEGRAM_MESSAGE_LENGTH = 4096


def split_long_message(text, max_length=MAX_TELEGRAM_MESSAGE_LENGTH, sep="\n"):
    chunks = []
    while len(text) > max_length:
        chunk = text[:max_length]
        if sep:
            # split the text on the last sep character, if it exists
            last_sep = chunk.rfind(sep)
            if last_sep != -1:
                chunk = chunk[: last_sep + 1]
        text = text[len(chunk) :]
        chunks.append(chunk)
    if text:
        chunks.append(text)  # add the remaining text as the last chunk
    return chunks


SPECIAL_CHARS = r"\\_\*\[\]\(\)~`><&#+\-=\|\{\}\.\!"


escape_re = re.compile(f"[{SPECIAL_CHARS}]")


def escape_md(text: str) -> str:
    """Escape markdown special characters in the text."""
    return escape_re.sub(r"\\\g<0>", text)


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 50


def split_text_with_overlap(
    text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """
    Splits a given text into smaller chunks using the RecursiveCharacterTextSplitter.

    This function is particularly useful for splitting large texts into smaller
    parts that are easier to manage and process, especially when dealing with
    language models that have token limits.

    Args:
        text (str): The text to be split into smaller chunks.
        chunk_size (int, optional): The size of each chunk in characters. Default is 1000.
        chunk_overlap (int, optional): The number of characters to overlap between chunks.
                                       Default is 50.

    Returns:
        list: A list of text chunks created from the original text.

    Example:
        >>> text = "This is a sample text that is longer than the chunk size."
        >>> split_text_with_overlap(text, chunk_size=10, chunk_overlap=5)
        ['This is a ', 'is a sample', 'ample text ', 'text that i', 'hat is long', 'is longer ', 'nger than ', 'han the chu', 'e chunk siz', 'chunk size.']
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)
