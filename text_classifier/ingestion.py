from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from text_classifier import config
from text_classifier.utils import clean_text, is_url, youtube_id

logger = logging.getLogger(__name__)
USER_AGENT = "Mozilla/5.0 (text-classifier; +https://github.com/mwi-kali/Text-Classifier)"
NOISE_TAGS = [
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "svg",
    "iframe",
]
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class IngestionError(ValueError):
    """Raised with a message that is safe to show to the user."""


@dataclass
class ExtractionOptions:
    include_lists: bool = True
    include_tables: bool = True
    include_code: bool = True
    min_block_words: int = 3
    max_chars: int = config.MAX_INPUT_CHARS
    render_js: bool | None = None
    allowed_tags: tuple[str, ...] = ()

    def tags(self) -> list[str]:
        if self.allowed_tags:
            return list(self.allowed_tags)
        tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "figcaption", "dd", "dt"]
        if self.include_lists:
            tags.append("li")
        if self.include_tables:
            tags.append("tr")
        if self.include_code:
            tags.append("pre")
        return tags


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class ExtractedDocument:
    text: str
    source: str
    source_type: str  # pdf | text | html | youtube
    title: str | None = None
    sections: list[Section] = field(default_factory=list)
    pages: int | None = None
    tables: int = 0
    rendered_js: bool = False

    @property
    def words(self) -> int:
        return len(self.text.split())

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d["headings"] = [s.heading for s in self.sections]
        d.pop("sections")
        d["words"] = self.words
        return d


def pdf_to_text(data: bytes) -> tuple[str, int]:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # one broken page should not lose the document
                pages.append("")
        return "\n\n".join(pages), len(reader.pages)
    except Exception as exc:
        raise IngestionError(f"Could not read that PDF ({exc.__class__.__name__}).") from exc


def decode_text(data: bytes) -> str:
    """UTF-8 (with or without BOM), UTF-16 only when a BOM says so, else Windows-1252."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def extract_bytes(
    data: bytes, filename: str, options: ExtractionOptions | None = None
) -> ExtractedDocument:
    options = options or ExtractionOptions()
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in config.SUPPORTED_FILE_EXTENSIONS:
        allowed = ", ".join(sorted(e.upper() for e in config.SUPPORTED_FILE_EXTENSIONS))
        raise IngestionError(f"Unsupported file type '.{ext}'. Use one of: {allowed}.")
    if ext == "pdf":
        if not data.lstrip()[:5].startswith(b"%PDF"):
            raise IngestionError("That file has a .pdf extension but is not a PDF.")
        raw, pages = pdf_to_text(data)
        return ExtractedDocument(clean_text(raw)[: options.max_chars], filename, "pdf", pages=pages)
    if ext in {"html", "htm"}:
        doc = html_to_document(decode_text(data), filename, options)
        return doc
    return ExtractedDocument(clean_text(decode_text(data))[: options.max_chars], filename, "text")


def bytes_to_text(data: bytes, filename: str) -> str:
    return extract_bytes(data, filename).text


def load_file(path: str | Path, options: ExtractionOptions | None = None) -> ExtractedDocument:
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    return extract_bytes(path.read_bytes(), str(path), options)


def _block_text(el: Tag) -> str:
    if el.name == "tr":
        cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
        return " | ".join(c for c in cells if c)
    if el.name == "li":
        return "• " + el.get_text(" ", strip=True)
    return el.get_text(" ", strip=True)


def html_to_document(
    html: str, source: str, options: ExtractionOptions | None = None
) -> ExtractedDocument:
    """Blocks in document order, grouped into sections under their nearest heading."""
    options = options or ExtractionOptions()
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    tags = options.tags()
    sections: list[Section] = [Section(title or "", "")]
    seen: set[str] = set()
    blocks: list[str] = []
    tables = len(soup.find_all("table"))
    for el in soup.find_all(tags):
        # skip blocks nested inside another selected block (e.g. <p> inside <li>) to avoid repeats
        if el.find_parent(tags) is not None and el.name not in HEADINGS:
            continue
        text = _block_text(el)
        if not text or text in seen:
            continue
        if el.name in HEADINGS:
            sections.append(Section(text, ""))
        elif len(text.split()) < options.min_block_words and el.name not in {"tr", "pre"}:
            continue
        else:
            sections[-1].text = f"{sections[-1].text} {text}".strip()
        seen.add(text)
        blocks.append(text)
    body = clean_text("\n".join(blocks))
    if not soup.find(["p", "h1", "h2", "h3", "li", "tr", "pre"]):  # fallback for <div>-only layouts
        fallback = clean_text((soup.body or soup).get_text(" ", strip=True))
        if len(fallback) > len(body):
            body = fallback
    sections = [s for s in sections if s.text]
    return ExtractedDocument(
        body[: options.max_chars], source, "html", title, sections, tables=tables
    )


def html_to_text(html: str) -> str:
    return html_to_document(html, "").text


def http_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers["User-Agent"] = USER_AGENT
    return s


_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    global _SESSION
    _SESSION = _SESSION or http_session()
    return _SESSION


BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)
# Text extraction never needs these, and skipping them makes ad-heavy pages load much faster.
BLOCKED_RESOURCES = {"image", "media", "font"}


def render_with_browser(url: str, timeout_ms: int = 20_000, settle_ms: int = 3_000) -> str:
    """Render a JavaScript-heavy page in headless Chromium (``browser`` extra).

    Waits for the DOM, then gives scripts up to ``settle_ms`` to fill it in. It deliberately
    does not wait for ``networkidle``: news and ad-funded sites poll trackers forever, so that
    state never arrives and every render would time out.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise IngestionError(
            "This page needs JavaScript to render. Install the browser extra: "
            "pip install -e '.[browser]' && playwright install chromium"
        ) from exc
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=BROWSER_USER_AGENT)
                page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type in BLOCKED_RESOURCES
                        else route.continue_()
                    ),
                )
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                # Late scripts may still be running; take whatever has rendered by then.
                with contextlib.suppress(PlaywrightError):
                    page.wait_for_load_state("load", timeout=settle_ms)
                return page.content()
            finally:
                browser.close()
    except PlaywrightError as exc:
        reason = "timed out" if "Timeout" in exc.__class__.__name__ else "failed"
        raise IngestionError(
            f"The headless browser {reason} loading that page. Try again with "
            "'Render JavaScript' turned off."
        ) from exc


def _playwright_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("playwright") is not None


def load_web(url: str, options: ExtractionOptions | None = None) -> ExtractedDocument:
    options = options or ExtractionOptions()
    if options.render_js:
        try:
            doc = html_to_document(render_with_browser(url), url, options)
        except IngestionError as exc:
            # A forced render that fails should not cost the user the page: most sites
            # still serve their text without JavaScript.
            logger.warning("%s; falling back to a static fetch of %s", exc, url)
        else:
            doc.rendered_js = True
            return doc
    resp = _session().get(url, timeout=(5, config.HTTP_TIMEOUT), stream=True)
    resp.raise_for_status()
    body = resp.raw.read(config.MAX_DOWNLOAD_BYTES + 1, decode_content=True)
    if len(body) > config.MAX_DOWNLOAD_BYTES:
        raise IngestionError("That page is larger than 10 MB.")
    ctype = resp.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        raw, pages = pdf_to_text(body)
        return ExtractedDocument(clean_text(raw)[: options.max_chars], url, "pdf", pages=pages)
    if "text/plain" in ctype:
        return ExtractedDocument(clean_text(decode_text(body))[: options.max_chars], url, "text")
    doc = html_to_document(body.decode(resp.encoding or "utf-8", errors="replace"), url, options)
    looks_js = doc.words < 20 and b"<script" in body.lower()
    if looks_js and options.render_js is None and _playwright_available():
        logger.info(
            "%s looks JavaScript-rendered (%d words); retrying in a headless browser",
            url,
            doc.words,
        )
        try:
            rendered = html_to_document(render_with_browser(url), url, options)
        except Exception as exc:  # browser missing or page broken: keep the static result
            logger.warning("headless render failed for %s: %s", url, exc)
            return doc
        if rendered.words > doc.words:
            rendered.rendered_js = True
            return rendered
    return doc


def load_youtube(video_id: str, url: str = "") -> ExtractedDocument:
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        parts = [snippet.text for snippet in YouTubeTranscriptApi().fetch(video_id)]
    except Exception as exc:
        raise IngestionError(
            f"No transcript available for that video ({exc.__class__.__name__})."
        ) from exc
    return ExtractedDocument(clean_text(" ".join(parts)), url or video_id, "youtube")


def load_url(url: str, options: ExtractionOptions | None = None) -> ExtractedDocument:
    if not is_url(url):
        raise IngestionError("Please enter a full http(s) URL.")
    vid = youtube_id(url)
    try:
        return load_youtube(vid, url) if vid else load_web(url, options)
    except requests.RequestException as exc:
        raise IngestionError(f"Could not fetch that URL ({exc.__class__.__name__}).") from exc


# --------------------------------------------------------------------------- entry points


def extract(
    source: str, is_url_source: bool | None = None, options: ExtractionOptions | None = None
) -> ExtractedDocument:
    """Dispatch on the source. Raises :class:`IngestionError` on anything unusable."""
    url = is_url(source) if is_url_source is None else is_url_source
    doc = load_url(source, options) if url else load_file(source, options)
    if not doc.text.strip():
        hint = " Scanned PDFs need OCR first." if doc.source_type == "pdf" else ""
        raise IngestionError(f"No readable text found.{hint}")
    return doc


def ingest(
    source: str, is_url_source: bool | None = None, options: ExtractionOptions | None = None
) -> str:
    return extract(source, is_url_source, options).text
