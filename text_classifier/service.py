from __future__ import annotations

import time

from text_classifier import config
from text_classifier.classification import Classifier
from text_classifier.ingestion import ExtractionOptions, IngestionError, extract
from text_classifier.schemas import AnalysisRequest, AnalysisResult
from text_classifier.summarization import Summarizer
from text_classifier.utils import clean_text


def analyze(
    request: AnalysisRequest,
    classifier: Classifier | None = None,
    summarizer: Summarizer | None = None,
) -> AnalysisResult:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    source = None
    if request.url:
        options = ExtractionOptions(
            include_lists=request.include_lists,
            include_tables=request.include_tables,
            include_code=request.include_code,
            max_chars=request.max_chars,
            render_js=request.render_js,
        )
        doc = extract(request.url, is_url_source=True, options=options)
        text, source = doc.text, doc.metadata()
    else:
        text = clean_text(request.text or "")
    if not text:
        raise IngestionError("No text to analyse.")
    text = text[: config.MAX_INPUT_CHARS]
    timings["ingest"] = round((time.perf_counter() - t0) * 1000, 1)

    result = AnalysisResult(characters=len(text), words=len(text.split()), source=source)
    if request.classify:
        t = time.perf_counter()
        result.classification = (classifier or Classifier()).classify(text, request.topics)
        timings["classify"] = round((time.perf_counter() - t) * 1000, 1)
    if request.summarize:
        t = time.perf_counter()
        s = summarizer or Summarizer()
        result.summary = (
            s.map_reduce(text) if request.summary_mode == "map-reduce" else s.abstractive(text)
        )
        timings["summarize"] = round((time.perf_counter() - t) * 1000, 1)
    result.timings_ms = timings
    return result
