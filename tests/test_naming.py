from diareader.naming import safe_filename


def test_strips_the_colon_that_started_this():
    # "Chương 2: Thanh Ngưu Trấn" is a real heading from a real book, and the colon
    # is what made Windows reject the name the app itself had proposed.
    assert safe_filename("Chương 2: Thanh Ngưu Trấn") == "Chương 2 Thanh Ngưu Trấn"


def test_keeps_vietnamese_diacritics():
    assert safe_filename("Quyển 1 - Nhập môn") == "Quyển 1 - Nhập môn"


def test_removes_every_character_windows_forbids():
    stem = safe_filename('a<b>c:d"e/f\\g|h?i*j')
    assert not set(stem) & set('<>:"/\\|?*')
    assert stem.startswith("a b c")


def test_collapses_the_gaps_left_behind():
    assert safe_filename("a :: b") == "a b"


def test_drops_trailing_dots_and_spaces_windows_would_eat():
    assert safe_filename("Chapter One. ") == "Chapter One"


def test_falls_back_when_nothing_survives():
    assert safe_filename("???") == "export"
    assert safe_filename("", fallback="book") == "book"


def test_sidesteps_reserved_device_names():
    # "CON.mp3" is not a file Windows will create, whatever the extension.
    assert safe_filename("CON") == "CON_"
    assert safe_filename("con") == "con_"
    assert safe_filename("CONtinued") == "CONtinued"


def test_caps_length_without_leaving_a_trailing_space():
    stem = safe_filename("word " * 200)
    assert len(stem) <= 120
    assert stem == stem.rstrip(" .")
