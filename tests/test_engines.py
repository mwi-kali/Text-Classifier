import pytest
from pydantic import ValidationError

from text_classifier.classification import Classifier
from text_classifier.schemas import AnalysisRequest
from text_classifier.service import analyze
from text_classifier.summarization import Summarizer, tidy_summary


def test_classifier_with_injected_pipelines(article, fakes):
    sent, zs, _ = fakes
    res = Classifier(sent, zs).classify(article)
    assert res.sentiment.label == "positive" and res.sentiment.score == 0.91
    assert res.style.label == "formal"
    assert len(res.topics) == 3
    topic_candidates = zs.calls[-1]
    assert topic_candidates[0] == "technology"  # suggested from the text, ahead of defaults
    assert len(topic_candidates) == len(set(topic_candidates))


def test_explicit_topics(article, fakes):
    sent, zs, _ = fakes
    res = Classifier(sent, zs).classify(article, topics=["fintech", "sport"], top_k=5)
    assert [t.label for t in res.topics] == ["fintech", "sport"]


def test_chunks_respect_budget_and_order(article, fakes):
    *_, summ = fakes
    s = Summarizer(summ, chunk_tokens=30)
    chunks = s.chunk(article * 3)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 30 for c in chunks)
    assert " ".join(chunks).split() == (article * 3).split()  # nothing lost or reordered


def test_hard_split_of_giant_sentence(fakes):
    *_, summ = fakes
    giant = " ".join(f"w{i}" for i in range(200))
    chunks = Summarizer(summ, chunk_tokens=50).chunk(giant)
    assert all(len(c.split()) <= 50 for c in chunks) and len(chunks) >= 4


def test_abstractive_keeps_document_order(article, fakes):
    *_, summ = fakes
    out = Summarizer(summ, chunk_tokens=30).abstractive(article)
    firsts = [c.split()[:3] for c in summ.calls]
    assert out.split() == [w for f in firsts for w in f]
    assert out.startswith("Nairobi's technology sector")


def test_map_reduce_converges(article, fakes):
    *_, summ = fakes
    out = Summarizer(summ, chunk_tokens=40).map_reduce(article * 4)
    assert 0 < len(out.split()) < len((article * 4).split())


def test_short_text_is_returned_unchanged(fakes):
    *_, summ = fakes
    assert Summarizer(summ).abstractive("Too short.") == "Too short."
    assert not summ.calls


def test_service(article, fakes):
    sent, zs, summ = fakes
    res = analyze(AnalysisRequest(text=article), Classifier(sent, zs), Summarizer(summ, 30))
    assert res.words == len(article.split())
    assert res.classification and res.summary
    assert set(res.timings_ms) == {"ingest", "classify", "summarize"}
    only = analyze(AnalysisRequest(text=article, classify=False, summarize=False))
    assert only.classification is None and only.summary is None


def test_request_needs_exactly_one_source():
    with pytest.raises(ValidationError):
        AnalysisRequest()
    with pytest.raises(ValidationError):
        AnalysisRequest(text="a", url="https://x.io")


@pytest.mark.models
def test_real_models_smoke(article):
    pytest.importorskip("transformers")
    res = analyze(AnalysisRequest(text=article * 2))
    assert res.classification.sentiment.label
    assert 0 < len(res.summary.split()) < len((article * 2).split())


def test_tidy_summary_fixes_spaced_punctuation():
    raw = " Trump suggests they did not agree before he announced it . Seven senators objected , too ."
    assert tidy_summary(raw) == (
        "Trump suggests they did not agree before he announced it. Seven senators objected, too."
    )
