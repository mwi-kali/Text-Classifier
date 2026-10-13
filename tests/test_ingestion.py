import io
import shutil

import pytest
import requests

from text_classifier import ingestion
from text_classifier.ingestion import (
    ExtractionOptions,
    IngestionError,
    bytes_to_text,
    extract,
    extract_bytes,
    html_to_document,
    ingest,
)

RICH_HTML = """<html><head><title>Release notes</title></head><body>
  <nav>Home About Contact us today please</nav>
  <h1>Version 2.0</h1>
  <p>This release rewrites the ingestion layer from scratch.</p>
  <p>This release rewrites the ingestion layer from scratch.</p>
  <h2>Changes</h2>
  <ul><li>Tables are now extracted row by row</li><li>Code blocks are preserved verbatim</li></ul>
  <table><tr><th>metric</th><th>value</th></tr><tr><td>p95</td><td>280 ms</td></tr></table>
  <pre>pip install text-classifier</pre>
  <figure><img src="x.png"><figcaption>Latency before and after the change</figcaption></figure>
  <footer>Copyright footer text here</footer>
</body></html>"""


def _pdf_bytes() -> bytes:
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_text_file_with_encoding_fallback(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_bytes("Ubunifu ni muhimu – café".encode("cp1252", errors="replace"))
    assert "Ubunifu" in ingest(str(p))


def test_html_file(tmp_path):
    p = tmp_path / "page.html"
    p.write_text(RICH_HTML)
    doc = extract(str(p))
    assert doc.source_type == "html" and doc.title == "Release notes"


def test_unsupported_extension():
    with pytest.raises(IngestionError, match="Unsupported"):
        bytes_to_text(b"x", "image.png")


def test_blank_pdf_reports_no_text(tmp_path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(_pdf_bytes())
    with pytest.raises(IngestionError, match="OCR"):
        ingest(str(p))


def test_pdf_metadata_counts_pages():
    doc = extract_bytes(_pdf_bytes(), "a.pdf")
    assert doc.pages == 1 and doc.source_type == "pdf"


def test_corrupt_and_mislabelled_pdf():
    with pytest.raises(IngestionError, match="PDF"):
        bytes_to_text(b"%PDF-garbage", "broken.pdf")
    with pytest.raises(IngestionError, match="not a PDF"):
        bytes_to_text(b"hello", "fake.pdf")


def test_missing_file():
    with pytest.raises(IngestionError, match="not found"):
        ingest("/no/such/file.txt")


def test_max_chars_limits_extraction_depth():
    doc = extract_bytes(b"word " * 5000, "big.txt", ExtractionOptions(max_chars=1000))
    assert len(doc.text) <= 1000


def test_rich_structure_is_captured():
    doc = html_to_document(RICH_HTML, "x")
    t = doc.text
    assert "• Tables are now extracted row by row" in t
    assert "metric | value" in t and "p95 | 280 ms" in t
    assert "pip install text-classifier" in t
    assert "Latency before and after" in t
    assert "Contact" not in t and "Copyright" not in t
    assert t.count("rewrites the ingestion layer") == 1
    assert doc.tables == 1


def test_sections_map_text_to_headings():
    doc = html_to_document(RICH_HTML, "x")
    headings = [s.heading for s in doc.sections]
    assert headings == ["Version 2.0", "Changes"]
    assert "Tables are now extracted" in doc.sections[1].text
    meta = doc.metadata()
    assert meta["headings"] == headings and meta["words"] == doc.words and "text" not in meta


def test_extraction_flags():
    lean = html_to_document(
        RICH_HTML,
        "x",
        ExtractionOptions(include_tables=False, include_lists=False, include_code=False),
    )
    assert "280 ms" not in lean.text and "• " not in lean.text and "pip install" not in lean.text
    only_h = html_to_document(RICH_HTML, "x", ExtractionOptions(allowed_tags=("h1", "h2")))
    assert only_h.text == "Version 2.0 Changes"


def test_div_only_pages_use_fallback_parser():
    html = (
        "<html><body><div>"
        + "Content living in plain divs without paragraphs. " * 3
        + "</div></body></html>"
    )
    assert "plain divs" in html_to_document(html, "x").text


class FakeResp:
    def __init__(self, body: bytes, ctype="text/html; charset=utf-8"):
        self.headers = {"Content-Type": ctype}
        self.raw = io.BytesIO(body)
        self.raw.read = lambda n, decode_content=True, _b=body: _b[:n]  # type: ignore[assignment]
        self.encoding = "utf-8"

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, resp):
        self.resp = resp

    def get(self, url, timeout, stream):
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def test_web_page(monkeypatch):
    monkeypatch.setattr(ingestion, "_SESSION", FakeSession(FakeResp(RICH_HTML.encode())))
    doc = extract("https://example.com/notes")
    assert doc.source_type == "html" and doc.title == "Release notes" and not doc.rendered_js


def test_web_plain_text_and_pdf_content_types(monkeypatch):
    monkeypatch.setattr(
        ingestion, "_SESSION", FakeSession(FakeResp(b"just   text here", "text/plain"))
    )
    assert extract("https://example.com/a").text == "just text here"
    monkeypatch.setattr(
        ingestion, "_SESSION", FakeSession(FakeResp(_pdf_bytes(), "application/pdf"))
    )
    with pytest.raises(IngestionError, match="OCR"):
        extract("https://example.com/report.pdf")


def test_oversized_page(monkeypatch):
    monkeypatch.setattr(ingestion.config, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(ingestion, "_SESSION", FakeSession(FakeResp(b"x" * 100)))
    with pytest.raises(IngestionError, match="larger than"):
        extract("https://example.com/huge")


def test_network_errors_become_friendly(monkeypatch):
    monkeypatch.setattr(ingestion, "_SESSION", FakeSession(requests.ConnectionError("down")))
    with pytest.raises(IngestionError, match="Could not fetch"):
        ingest("https://example.com")


def test_malformed_url():
    with pytest.raises(IngestionError, match="full http"):
        ingestion.load_url("example.com/no-scheme")


def test_retries_and_timeouts_are_configured():
    retry = ingestion.http_session(retries=5).get_adapter("https://x.io").max_retries
    assert retry.total == 5 and {429, 503} <= set(retry.status_forcelist)


def test_js_heavy_page_falls_back_to_browser(monkeypatch):
    shell = b"<html><body><div id='app'></div><script>render()</script></body></html>"
    monkeypatch.setattr(ingestion, "_SESSION", FakeSession(FakeResp(shell)))
    monkeypatch.setattr(ingestion, "_playwright_available", lambda: True)
    monkeypatch.setattr(ingestion, "render_with_browser", lambda url: RICH_HTML)
    doc = extract("https://spa.example.com")
    assert doc.rendered_js and "Version 2.0" in doc.text


def test_render_js_flag_forces_browser(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ingestion, "render_with_browser", lambda url: calls.append(url) or RICH_HTML
    )
    doc = extract("https://example.com", options=ExtractionOptions(render_js=True))
    assert calls == ["https://example.com"] and doc.rendered_js


def test_failed_forced_render_falls_back_to_static_fetch(monkeypatch):
    # e.g. a news site whose trackers keep the browser busy until it times out
    def timed_out(url):
        raise IngestionError("The headless browser timed out loading that page.")

    monkeypatch.setattr(ingestion, "render_with_browser", timed_out)
    monkeypatch.setattr(ingestion, "_SESSION", FakeSession(FakeResp(RICH_HTML.encode())))
    doc = extract("https://news.example.com", options=ExtractionOptions(render_js=True))
    assert "Version 2.0" in doc.text and not doc.rendered_js


@pytest.mark.browser
def test_real_headless_render(tmp_path):
    pytest.importorskip("playwright")
    if not shutil.which("chromium") and not __import__("os").environ.get(
        "PLAYWRIGHT_BROWSERS_PATH"
    ):
        pytest.skip("no Chromium installed for Playwright")
    page = tmp_path / "spa.html"
    page.write_text(
        "<html><body><div id='app'></div><script>document.getElementById('app').innerHTML="
        "'<h1>Rendered</h1><p>Only visible after JavaScript runs in a browser.</p>';</script></body></html>"
    )
    doc = html_to_document(ingestion.render_with_browser(page.as_uri()), "spa")
    assert "Only visible after JavaScript" in doc.text


# ---------------------------------------------------------------- YouTube
def test_youtube_dispatch_and_failure(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "load_youtube",
        lambda vid, url="": ingestion.ExtractedDocument(f"transcript of {vid}", url, "youtube"),
    )
    doc = extract("https://youtu.be/dQw4w9WgXcQ")
    assert doc.text == "transcript of dQw4w9WgXcQ" and doc.source_type == "youtube"


def test_youtube_without_transcript(monkeypatch):
    import sys
    import types

    class Api:
        def fetch(self, vid):
            raise RuntimeError("TranscriptsDisabled")

    monkeypatch.setitem(
        sys.modules, "youtube_transcript_api", types.SimpleNamespace(YouTubeTranscriptApi=Api)
    )
    with pytest.raises(IngestionError, match="No transcript"):
        ingestion.load_youtube("dQw4w9WgXcQ")
