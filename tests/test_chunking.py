from diareader.chunking import (
    chunk_for_narration, chunk_text, is_speakable, split_sentences,
)


def test_stray_punctuation_is_not_a_sentence():
    """A lone quote mark stranded on its own line has no audio to produce, but the
    pipeline used to spend the full retry budget discovering that."""
    assert not is_speakable("“")
    assert not is_speakable("...")
    assert not is_speakable("— —")
    assert is_speakable("Ừ.")
    assert is_speakable("A1")


def test_split_sentences_drops_unspeakable_fragments():
    text = 'Hắn nói.\n“\nCô ấy trả lời.'
    assert split_sentences(text) == ["Hắn nói.", "Cô ấy trả lời."]


def test_split_sentences_basic():
    text = "Xin chào. Hôm nay trời đẹp! Bạn khỏe không?"
    sentences = split_sentences(text)
    assert sentences == ["Xin chào.", "Hôm nay trời đẹp!", "Bạn khỏe không?"]


def test_split_sentences_multi_paragraph():
    text = "Câu một. Câu hai.\nĐoạn mới ở đây."
    sentences = split_sentences(text)
    assert sentences == ["Câu một.", "Câu hai.", "Đoạn mới ở đây."]


def test_chunk_text_respects_char_budget():
    sentence = "Đây là một câu ví dụ khá dài để kiểm tra việc chia đoạn văn bản."
    text = " ".join([sentence] * 10)
    chunks = chunk_text(text, max_chars=100)
    assert all(len(c) <= 100 or c.strip() == sentence for c in chunks)
    assert "".join(chunks).replace(" ", "") != ""


def test_chunk_text_emits_one_sentence_per_chunk():
    # The checkpoint stops at the first terminal period and drops the rest, so
    # sentences must never be grouped together — even when they'd fit the budget.
    text = "Một. Hai. Ba. Bốn. Năm."
    assert chunk_text(text, max_chars=1000) == ["Một.", "Hai.", "Ba.", "Bốn.", "Năm."]


def test_chunk_text_splits_oversized_sentence_at_commas():
    long_sentence = "Phần một rất dài, phần hai cũng dài, phần ba kết thúc."
    chunks = chunk_text(long_sentence, max_chars=25)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)
    # every chunk boundary lands after a comma, never inside a word
    assert " ".join(chunks) == long_sentence


def test_chunk_text_keeps_unsplittable_sentence_whole():
    # No commas to break on: better to send it long than to cut mid-word.
    long_sentence = "Từ " * 100 + "kết thúc."
    assert chunk_text(long_sentence, max_chars=50) == [long_sentence]


def test_chunk_for_narration_emits_untagged_text():
    # The Vietnamese fine-tune was trained on plain text with no [S1]/[S2] tags;
    # feeding it tagged input is out-of-distribution and makes it repeat fragments.
    chunks = chunk_for_narration("Xin chào. Tạm biệt.")
    assert chunks == ["Xin chào.", "Tạm biệt."]
    assert not any("[S" in c for c in chunks)


def test_wrap_for_narration_applies_tag_only_when_set():
    from diareader.chunking import wrap_for_narration

    assert wrap_for_narration("Xin chào.") == "Xin chào."
    assert wrap_for_narration("Xin chào.", tag="[S1]") == "[S1] Xin chào."


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   \n  \n") == []
