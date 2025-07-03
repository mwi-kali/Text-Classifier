import os
import requests

from bs4 import BeautifulSoup
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_unstructured import UnstructuredLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
from text_classifier.utils import clean_text


class Ingestor:
    @staticmethod
    def _load_file(path: str) -> str:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext == "pdf":
            loader = PyMuPDF4LLMLoader(path)
        elif ext in ("txt", "md"):
            loader = UnstructuredLoader(
                [path],
                chunking_strategy="basic",
                max_characters=10**6,
                include_orig_elements=False,
            )
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        docs = loader.load()
        raw = "\n\n".join(d.page_content for d in docs)
        return clean_text(raw)

    @staticmethod
    def _load_url(url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            loader = YoutubeLoaderDL.from_youtube_url(url, add_video_info=False)
            docs = loader.load()
            raw = "\n\n".join(d.page_content for d in docs)
            return clean_text(raw)

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        pieces = []

        if soup.title and soup.title.string:
            pieces.append(soup.title.string.strip())

        for level in range(1, 7):
            for tag in soup.find_all(f"h{level}"):
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    pieces.append(text)

        for p in soup.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text:
                pieces.append(text)

        raw = "\n\n".join(pieces)
        return clean_text(raw)

    def ingest(self, source: str, is_url: bool = False) -> str:
        if is_url:
            return self._load_url(source)
        else:
            return self._load_file(source)
