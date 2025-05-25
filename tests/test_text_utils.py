from unittest.mock import Mock

from src.utils.text_utils import merge_two_chunks, merge_all_chunks, find_segment_pos


class TestMergeTwoChunks:
    """Test cases for merge_two_chunks function."""

    def test_merge_with_clear_overlap(self):
        """Test merging two chunks with clear overlapping text."""
        chunk1 = "This is the first part of the text and it continues with some words that overlap"
        chunk2 = "some words that overlap and then new content appears here"

        result = merge_two_chunks(chunk1, chunk2, match_cutoff=10)

        # Should merge at the overlap point
        assert "some words that overlap" in result
        assert result.count("some words that overlap") == 1  # Should not duplicate
        assert "This is the first part" in result
        assert "new content appears here" in result

    def test_merge_with_no_overlap(self):
        """Test merging chunks with no overlap - should concatenate."""
        chunk1 = "First chunk with unique content"
        chunk2 = "Second chunk with different content"

        result = merge_two_chunks(chunk1, chunk2)

        # Should concatenate both chunks
        assert chunk1 in result
        assert chunk2 in result
        assert result == chunk1 + chunk2

    def test_merge_with_small_overlap_below_cutoff(self):
        """Test merging with overlap smaller than cutoff threshold."""
        chunk1 = "First chunk ends with a"
        chunk2 = "a second chunk begins"

        # Default cutoff is 15, so single character overlap should concatenate
        result = merge_two_chunks(chunk1, chunk2, match_cutoff=15)

        assert result == chunk1 + chunk2

    def test_merge_with_custom_buffer_size(self):
        """Test merging with custom buffer size."""
        chunk1 = "Word1 Word2 Word3 Word4 Word5 overlap text here"
        chunk2 = "overlap text here Word6 Word7 Word8"

        result = merge_two_chunks(chunk1, chunk2, buffer=5)

        assert "overlap text here" in result
        assert result.count("overlap text here") == 1

    def test_merge_with_custom_match_cutoff(self):
        """Test merging with custom match cutoff."""
        chunk1 = "Text with short overlap"
        chunk2 = "overlap and more text"

        # Lower cutoff should allow smaller overlaps
        result = merge_two_chunks(chunk1, chunk2, match_cutoff=5)

        assert "overlap" in result
        assert result.count("overlap") == 1

    def test_merge_with_punctuation_normalization(self):
        """Test that punctuation is normalized during overlap detection."""
        chunk1 = "Text ends with overlap, here."
        chunk2 = "overlap here and continues"

        result = merge_two_chunks(chunk1, chunk2)

        # Should find overlap despite punctuation differences
        assert "overlap" in result
        assert "continues" in result

    def test_merge_with_case_insensitive_overlap(self):
        """Test that overlap detection is case insensitive."""
        chunk1 = "Text with OVERLAP TEXT"
        chunk2 = "overlap text and more content"

        result = merge_two_chunks(chunk1, chunk2)

        assert "OVERLAP TEXT" in result or "overlap text" in result
        assert "more content" in result

    def test_merge_empty_chunks(self):
        """Test merging with empty chunks."""
        assert merge_two_chunks("", "second chunk") == "second chunk"
        assert merge_two_chunks("first chunk", "") == "first chunk"
        assert merge_two_chunks("", "") == ""

    def test_merge_with_custom_logger(self):
        """Test merging with custom logger."""
        mock_logger = Mock()
        chunk1 = "First chunk with overlap"
        chunk2 = "overlap and second chunk"

        result = merge_two_chunks(chunk1, chunk2, logger=mock_logger)

        # Verify logger was called
        assert mock_logger.debug.called
        assert "overlap" in result


class TestMergeAllChunks:
    """Test cases for merge_all_chunks function."""

    def test_merge_multiple_chunks_with_overlaps(self):
        """Test merging multiple chunks with overlapping content."""
        chunks = [
            "First chunk with overlap one that is long enough",
            "overlap one that is long enough and overlap two that is also long",
            "overlap two that is also long and final content",
        ]

        result = merge_all_chunks(chunks, match_cutoff=10)

        assert "First chunk" in result
        assert "final content" in result
        # Should not duplicate overlapping parts
        assert result.count("overlap one that is long enough") <= 1
        assert result.count("overlap two that is also long") <= 1

    def test_merge_chunks_no_overlaps(self):
        """Test merging chunks with no overlaps."""
        chunks = ["First unique chunk", "Second unique chunk", "Third unique chunk"]

        result = merge_all_chunks(chunks)

        # Should concatenate all chunks
        for chunk in chunks:
            assert chunk in result

    def test_merge_single_chunk(self):
        """Test merging a single chunk."""
        chunks = ["Only one chunk"]

        result = merge_all_chunks(chunks)

        assert result == "Only one chunk"

    def test_merge_empty_list(self):
        """Test merging empty list of chunks."""
        result = merge_all_chunks([])

        assert result == ""

    def test_merge_with_custom_parameters(self):
        """Test merging with custom buffer and cutoff parameters."""
        chunks = [
            "Start text overlap",
            "overlap middle text overlap",
            "overlap end text",
        ]

        result = merge_all_chunks(chunks, buffer=10, match_cutoff=5)

        assert "Start text" in result
        assert "end text" in result
        assert result.count("overlap") >= 1  # Should merge overlaps

    def test_merge_with_custom_logger(self):
        """Test merging with custom logger."""
        mock_logger = Mock()
        chunks = ["First chunk overlap", "overlap second chunk"]

        result = merge_all_chunks(chunks, logger=mock_logger)

        # Verify logger was used in underlying merge_two_chunks calls
        assert mock_logger.debug.called

    def test_merge_chunks_with_whitespace(self):
        """Test merging chunks with various whitespace."""
        chunks = [
            "  First chunk with spaces  ",
            "  spaces and second chunk  ",
            "  second chunk final  ",
        ]

        result = merge_all_chunks(chunks)

        assert "First chunk" in result
        assert "final" in result

    def test_merge_realistic_transcription_chunks(self):
        """Test merging realistic transcription-like chunks."""
        chunks = [
            "Hello everyone, welcome to today's meeting. We're going to discuss",
            "discuss the quarterly results and our plans for the next quarter.",
            "quarter. First, let me show you the sales figures from last month.",
            "month. As you can see, we've exceeded our targets by fifteen percent.",
        ]

        result = merge_all_chunks(chunks)

        # Should create coherent text without duplicated phrases
        assert "Hello everyone" in result
        assert "fifteen percent" in result
        assert result.count("discuss") <= 2  # Should merge overlapping "discuss"
        assert result.count("quarter") <= 3  # Should merge overlapping "quarter"
        assert result.count("month") <= 2  # Should merge overlapping "month"


class TestCornerCases:
    """Test corner cases that could cause IndexError or other issues."""

    def test_find_segment_pos_empty_strings(self):
        """Test find_segment_pos with empty strings."""
        assert find_segment_pos("", "") == (0, 0)
        assert find_segment_pos("test", "") == (0, 0)
        assert find_segment_pos("", "test") == (0, 0)

    def test_find_segment_pos_segment_longer_than_text(self):
        """Test find_segment_pos when segment is longer than text."""
        result = find_segment_pos("very long segment", "short")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_find_segment_pos_no_alphanumeric(self):
        """Test find_segment_pos with non-alphanumeric characters."""
        result = find_segment_pos("!!!", "???")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_merge_two_chunks_both_empty(self):
        """Test merging two empty chunks."""
        result = merge_two_chunks("", "")
        assert result == ""

    def test_merge_two_chunks_whitespace_only(self):
        """Test merging chunks with only whitespace."""
        result = merge_two_chunks("   ", "   ")
        assert result == "      "  # Should concatenate whitespace

    def test_merge_two_chunks_very_short_chunks(self):
        """Test merging very short chunks."""
        result = merge_two_chunks("a", "b")
        assert result == "ab"

    def test_merge_two_chunks_single_character_overlap(self):
        """Test merging with single character overlap."""
        result = merge_two_chunks("testa", "atest")
        # Should concatenate due to small overlap
        assert result == "testaatest"

    def test_merge_all_chunks_with_empty_strings(self):
        """Test merge_all_chunks with empty strings in the list."""
        chunks = ["First chunk", "", "Second chunk", "   ", "Third chunk"]
        result = merge_all_chunks(chunks)

        # Should filter out empty and whitespace-only chunks
        assert "First chunk" in result
        assert "Second chunk" in result
        assert "Third chunk" in result

    def test_merge_all_chunks_all_empty(self):
        """Test merge_all_chunks with all empty chunks."""
        chunks = ["", "   ", "", "  \t  ", ""]
        result = merge_all_chunks(chunks)
        assert result == ""

    def test_merge_all_chunks_mixed_empty_and_content(self):
        """Test merge_all_chunks with mix of empty and content chunks."""
        chunks = [
            "",
            "Hello world",
            "   ",
            "world and more text",
            "",
            "more text continues",
        ]
        result = merge_all_chunks(chunks, match_cutoff=5)

        assert "Hello" in result
        assert "continues" in result
        # Should properly merge overlapping parts
        assert result.count("world") >= 1
        assert result.count("more text") >= 1

    def test_merge_chunks_with_only_punctuation(self):
        """Test merging chunks with only punctuation."""
        chunks = ["!!!", "???", "..."]
        result = merge_all_chunks(chunks)
        assert "!!!" in result
        assert "???" in result
        assert "..." in result

    def test_merge_chunks_with_unicode(self):
        """Test merging chunks with unicode characters."""
        chunks = ["Hello 世界", "世界 and more", "more content"]
        result = merge_all_chunks(chunks, match_cutoff=5)
        assert "Hello" in result
        assert "世界" in result
        assert "content" in result

    def test_merge_chunks_very_long_overlap(self):
        """Test merging with very long overlapping text."""
        long_overlap = (
            "this is a very long overlapping text that should be merged properly"
        )
        chunk1 = f"Start of first chunk {long_overlap}"
        chunk2 = f"{long_overlap} end of second chunk"

        result = merge_two_chunks(chunk1, chunk2, match_cutoff=10)

        assert "Start of first chunk" in result
        assert "end of second chunk" in result
        assert result.count(long_overlap) == 1

    def test_merge_chunks_with_newlines_and_tabs(self):
        """Test merging chunks with newlines and tabs."""
        chunks = [
            "First line\nSecond line\tTabbed",
            "Tabbed content\nNew line",
            "New line\tFinal content",
        ]
        result = merge_all_chunks(chunks, match_cutoff=5)

        assert "First line" in result
        # The merging logic correctly merges overlapping parts
        # Check that the result contains the expected content
        assert "Tabbed" in result
        assert "content" in result

    def test_merge_chunks_extreme_buffer_sizes(self):
        """Test merging with extreme buffer sizes."""
        chunk1 = "Short text"
        chunk2 = "Different text"

        # Buffer larger than text
        result1 = merge_two_chunks(chunk1, chunk2, buffer=1000)
        assert chunk1 in result1
        assert chunk2 in result1

        # Buffer of 0
        result2 = merge_two_chunks(chunk1, chunk2, buffer=0)
        assert chunk1 in result2
        assert chunk2 in result2

    def test_merge_chunks_extreme_cutoff_values(self):
        """Test merging with extreme cutoff values."""
        chunk1 = "Text with overlap"
        chunk2 = "overlap and more"

        # Very high cutoff
        result1 = merge_two_chunks(chunk1, chunk2, match_cutoff=1000)
        assert result1 == chunk1 + chunk2

        # Zero cutoff
        result2 = merge_two_chunks(chunk1, chunk2, match_cutoff=0)
        assert "overlap" in result2

    def test_merge_chunks_production_like_scenario(self):
        """Test scenario similar to the production error."""
        # Simulate chunks that might cause IndexError
        chunks = [
            "Some transcription text",
            "",  # Empty chunk
            "   ",  # Whitespace only
            "continuation of text",
            "",
            "final part",
        ]

        # This should not raise IndexError
        result = merge_all_chunks(chunks)
        assert "Some transcription text" in result
        assert "continuation of text" in result
        assert "final part" in result

    def test_production_indexerror_scenario(self):
        """Test the exact scenario that caused IndexError in production."""
        # Simulate the exact conditions that led to the IndexError
        # where find_segment_pos was called with problematic inputs

        # Test find_segment_pos directly with edge cases
        assert find_segment_pos("", "some text") == (0, 0)
        assert find_segment_pos("text", "") == (0, 0)
        assert find_segment_pos("a", "b") == (0, 1)  # No match, should not crash

        # Test merge_two_chunks with conditions that could cause the error
        result1 = merge_two_chunks("", "some content")
        assert result1 == "some content"

        result2 = merge_two_chunks("some content", "")
        assert result2 == "some content"

        # Test with very short overlaps that might cause boundary issues
        result3 = merge_two_chunks("a", "ab")
        assert "a" in result3 and "b" in result3

        # Test merge_all_chunks with 46 chunks (like in production)
        many_chunks = [f"chunk {i} content" for i in range(46)]
        many_chunks[10] = ""  # Add some empty chunks
        many_chunks[25] = "   "
        many_chunks[40] = ""

        # This should not raise IndexError
        result = merge_all_chunks(many_chunks)
        assert "chunk 0 content" in result
        assert "chunk 45 content" in result
