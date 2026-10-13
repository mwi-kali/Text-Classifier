import pytest

ARTICLE = (
    "Nairobi's technology sector keeps growing. Startups in Nairobi are building payment "
    "technology for small businesses across Kenya. Investors say the technology ecosystem "
    "has matured, and several startups have raised large rounds this year. Analysts expect "
    "payment startups to expand into neighbouring markets. Regulators are drafting new rules "
    "for digital lenders, which startups say could slow growth. Still, the mood among founders "
    "is optimistic and hiring remains strong across the technology industry."
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeSentiment:
    def __call__(self, text, truncation=True):
        return [{"label": "Positive", "score": 0.91}]


class FakeZeroShot:
    def __init__(self):
        self.calls = []

    def __call__(self, text, candidate_labels, multi_label=False):
        self.calls.append(list(candidate_labels))
        n = len(candidate_labels)
        scores = [round(1 - i / (n + 1), 3) for i in range(n)]
        return {"labels": list(candidate_labels), "scores": scores}


class FakeSummarizer:
    """Returns the first few words of each chunk so ordering is checkable."""

    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def __call__(self, texts, max_length, min_length, do_sample, truncation):
        assert min_length < max_length and not do_sample
        batch = [texts] if isinstance(texts, str) else list(texts)
        self.calls.extend(batch)
        self.batches = getattr(self, "batches", []) + [len(batch)]
        return [{"summary_text": " ".join(t.split()[:3])} for t in batch]


@pytest.fixture
def article():
    return ARTICLE


@pytest.fixture
def fakes():
    return FakeSentiment(), FakeZeroShot(), FakeSummarizer()
