import logging
import spacy
import re

from bs4 import BeautifulSoup
from spacy.cli import download as spacy_download
from urllib.parse import urlparse


def clean_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_url(text: str) -> bool:
    try:
        parts = urlparse(text)
        return parts.scheme in ("http", "https") and bool(parts.netloc)
    except Exception:
        return False


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=level,
    )


def preprocess_text(
    text: str,
    lower: bool = False,
    use_spacy: bool = False
) -> str:
    try:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        raw = soup.get_text(separator=" ")

        if use_spacy:
            
            try:
                nlp = spacy.load("xx_ent_wiki_sm")
            except OSError:
                logging.warning("Downloading spaCy model xx_ent_wiki_sm…")
                spacy_download("xx_ent_wiki_sm")
                nlp = spacy.load("xx_ent_wiki_sm")
            sentences = []
            for sent in nlp(raw).sents:
                s = re.sub(r"\[.*?\]", "", sent.text)
                s = re.sub(r"\s+", " ", s)
                if lower:
                    s = s.lower()
                sentences.append(s.strip())
            cleaned = " ".join(sentences)
        else:
            cleaned = re.sub(r"\[.*?\]", "", raw)
            cleaned = re.sub(r"\s+", " ", cleaned)
            cleaned = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", cleaned)
            if lower:
                cleaned = cleaned.lower()
            cleaned = cleaned.strip()
        return cleaned
    except Exception as e:
        logging.error(f"Error during text preprocessing: {e}")
        return text.strip()
