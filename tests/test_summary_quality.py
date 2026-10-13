from pathlib import Path

import pytest

from text_classifier.evaluation import (
    evaluate,
    flesch_reading_ease,
    load_corpus,
    rouge_l,
    to_markdown,
)
from text_classifier.summarization import Summarizer

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus.jsonl"


def test_rouge_l():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert rouge_l("the cat sat", "dogs bark loudly") == 0.0
    assert 0 < rouge_l("the cat sat on the mat", "the cat lay on a mat") < 1


def test_reading_ease_orders_simple_above_dense():
    simple = "The cat sat. The dog ran. We had fun."
    dense = "Institutional heterogeneity complicates organisational interoperability considerations substantially."
    assert flesch_reading_ease(simple) > flesch_reading_ease(dense)


def test_chunks_overlap(article, fakes):
    *_, summ = fakes
    text = article * 2
    no_overlap = Summarizer(summ, chunk_tokens=60, overlap_tokens=0).chunk(text)
    chunks = Summarizer(summ, chunk_tokens=60, overlap_tokens=30).chunk(text)
    assert sum(len(c.split()) for c in no_overlap) == len(text.split())  # exact partition
    assert sum(len(c.split()) for c in chunks) > len(text.split())  # sentences repeated as context
    assert any(b.split(". ")[0] in a for a, b in zip(chunks, chunks[1:], strict=False))
    assert all(len(c.split()) <= 60 for c in chunks)


def test_invalid_overlap(fakes):
    *_, summ = fakes
    with pytest.raises(ValueError):
        Summarizer(summ, chunk_tokens=20, overlap_tokens=20)


def test_batching(article, fakes):
    *_, summ = fakes
    s = Summarizer(summ, chunk_tokens=30, overlap_tokens=0, batch_size=3)
    s.abstractive(article * 2)
    assert max(summ.batches) <= 3 and s.calls == len(summ.batches)
    assert sum(summ.batches) == len(s.chunk(article * 2))


def test_tree_reduce_never_feeds_oversized_inputs(article, fakes):
    *_, summ = fakes
    s = Summarizer(summ, chunk_tokens=40, overlap_tokens=0)
    s.map_reduce(article * 10)
    assert all(len(c.split()) <= 40 for c in summ.calls)


def test_evaluation_harness_on_corpus(fakes):
    *_, summ = fakes
    corpus = load_corpus(CORPUS)
    assert len(corpus) == 3 and all({"id", "text", "reference"} <= set(c) for c in corpus)
    rows = evaluate(Summarizer(summ, chunk_tokens=60), corpus)
    assert len(rows) == 6
    assert all(
        0 <= r.rouge_l <= 1 and 0 < r.compression < 1 and r.forward_passes >= 1 for r in rows
    )
    assert to_markdown(rows).startswith("| doc_id | mode | rouge_l")


@pytest.mark.models
def test_real_model_quality():
    pytest.importorskip("transformers")
    rows = evaluate(Summarizer(), load_corpus(CORPUS), modes=("abstractive",))
    assert sum(r.rouge_l for r in rows) / len(rows) > 0.15
    assert all(r.compression < 0.8 for r in rows)
