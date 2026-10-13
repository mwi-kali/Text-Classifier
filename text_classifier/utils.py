from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", level=level)


def strip_html(raw: str) -> str:
    if "<" not in raw:
        return raw
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def clean_text(raw: str) -> str:
    """
    Remove markup, bracketed citations like [12] and repeated whitespace.

    Unlike the first version this keeps non-ASCII letters, so French or Swahili text is not
    shredded before the multilingual sentiment model sees it.
    """
    text = strip_html(raw)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[​﻿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_url(text: str) -> bool:
    try:
        parts = urlparse(text.strip())
    except (ValueError, AttributeError):
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def youtube_id(url: str) -> str | None:
    if "youtube.com" not in url and "youtu.be" not in url:
        return None
    m = _YOUTUBE_ID.search(url)
    return m.group(1) if m else None


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def suggest_topics(text: str, top_k: int = 5) -> list[str]:
    """
    Frequent content words as candidate topic labels for zero-shot classification.

    The first version loaded a multilingual spaCy NER model (and pip-installed it at runtime)
    for this; a stop-word-filtered frequency count is lighter and works as well for proposing
    candidates, since the zero-shot model does the actual ranking.
    """
    words = [w.lower().strip("-'") for w in _WORD.findall(text)]
    counts = Counter(w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 3)
    return [w for w, c in counts.most_common(top_k) if c > 1]
