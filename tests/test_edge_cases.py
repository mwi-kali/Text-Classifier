import pytest
from pydantic import ValidationError

from text_classifier import config
from text_classifier.classification import Classifier
from text_classifier.ingestion import IngestionError
from text_classifier.schemas import AnalysisRequest
from text_classifier.service import analyze
from text_classifier.summarization import Summarizer


def test_empty_and_whitespace_inputs(fakes):
    sent, zs, summ = fakes
    with pytest.raises(ValidationError):
        AnalysisRequest(text="")
    with pytest.raises(IngestionError, match="No text"):
        analyze(AnalysisRequest(text="   <p> </p> "), Classifier(sent, zs), Summarizer(summ))


def test_oversized_text_is_rejected_by_the_api_contract():
    with pytest.raises(ValidationError):
        AnalysisRequest(text="x" * (config.MAX_INPUT_CHARS + 1))


def test_large_text_end_to_end(article, fakes):
    sent, zs, summ = fakes
    big = article * 200  # ~14k words
    res = analyze(
        AnalysisRequest(text=big, summary_mode="map-reduce"),
        Classifier(sent, zs),
        Summarizer(summ, 120),
    )
    assert res.words == len(big.split())
    assert 0 < len(res.summary.split()) < res.words / 10


def test_file_to_classification_to_summary(tmp_path, article, fakes):
    from text_classifier.ingestion import extract

    sent, zs, summ = fakes
    p = tmp_path / "story.md"
    p.write_text("# Heading\n\n" + article)
    text = extract(str(p)).text
    res = analyze(
        AnalysisRequest(text=text, topics=["fintech", "farming"]),
        Classifier(sent, zs),
        Summarizer(summ, 40),
    )
    assert [t.label for t in res.classification.topics] == ["fintech", "farming"]
    assert res.summary


def test_url_flow_reports_source_metadata(monkeypatch, fakes):
    import text_classifier.service as service
    from text_classifier.ingestion import ExtractedDocument, Section

    sent, zs, summ = fakes
    doc = ExtractedDocument(
        "Long enough text about Nairobi startups and payments. " * 10,
        "https://x.io",
        "html",
        title="Startups",
        sections=[Section("Intro", "...")],
    )
    seen = {}

    def fake_extract(url, is_url_source, options):
        seen["options"] = options
        return doc

    monkeypatch.setattr(service, "extract", fake_extract)
    res = analyze(
        AnalysisRequest(url="https://x.io", include_tables=False, render_js=True),
        Classifier(sent, zs),
        Summarizer(summ),
    )
    assert res.source["title"] == "Startups" and res.source["headings"] == ["Intro"]
    assert seen["options"].include_tables is False and seen["options"].render_js is True


def test_malformed_url_request(fakes):
    sent, zs, summ = fakes
    with pytest.raises(IngestionError, match="full http"):
        analyze(AnalysisRequest(url="not a url"), Classifier(sent, zs), Summarizer(summ))
