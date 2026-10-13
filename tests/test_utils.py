import pytest

from text_classifier.utils import clean_text, first_words, is_url, suggest_topics, youtube_id


def test_clean_text_strips_markup_and_keeps_unicode():
    raw = "<p>Habari  ya <b>leo</b>?</p><script>x()</script> Café [12] déjà"
    assert clean_text(raw) == "Habari ya leo ? Café déjà"


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("https://example.com/a", True),
        ("http://x.io", True),
        ("ftp://x.io", False),
        ("example.com", False),
        ("", False),
    ],
)
def test_is_url(value, ok):
    assert is_url(value) is ok


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=3",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    ],
)
def test_youtube_id(url):
    assert youtube_id(url) == "dQw4w9WgXcQ"


def test_youtube_id_none_for_other_sites():
    assert youtube_id("https://example.com/watch?v=dQw4w9WgXcQ") is None


def test_suggest_topics(article):
    topics = suggest_topics(article)
    assert topics[0] == "technology"
    assert "startups" in topics and "the" not in topics


def test_first_words():
    assert first_words("a b c d", 2) == "a b"
