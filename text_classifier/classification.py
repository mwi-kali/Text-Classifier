from __future__ import annotations

import logging
from collections.abc import Callable

from text_classifier import config, models
from text_classifier.schemas import ClassificationResult, LabelScore
from text_classifier.utils import first_words, suggest_topics

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError:
        return "unknown"
    DetectorFactory.seed = 0
    try:
        return detect(text) if text.strip() else "unknown"
    except LangDetectException:
        return "unknown"


class Classifier:
    """Pipelines are injected so tests (and alternative backends) don't need model downloads."""

    def __init__(
        self,
        sentiment_pipe: Callable | None = None,
        zero_shot_pipe: Callable | None = None,
        default_topics: list[str] | None = None,
    ) -> None:
        self._sentiment_pipe = sentiment_pipe
        self._zero_shot_pipe = zero_shot_pipe
        self.default_topics = default_topics or config.DEFAULT_TOPICS

    @property
    def sentiment_pipe(self):
        return self._sentiment_pipe if self._sentiment_pipe is not None else models.sentiment()

    @property
    def zero_shot_pipe(self):
        return self._zero_shot_pipe if self._zero_shot_pipe is not None else models.zero_shot()

    def _rank(self, text: str, labels: list[str], multi_label: bool = False) -> list[LabelScore]:
        res = self.zero_shot_pipe(text, candidate_labels=labels, multi_label=multi_label)
        return [
            LabelScore(label=lab, score=round(float(s), 4))
            for lab, s in zip(res["labels"], res["scores"], strict=True)
        ]

    def classify(
        self, text: str, topics: list[str] | None = None, top_k: int = 3
    ) -> ClassificationResult:
        window = first_words(text, config.CLASSIFY_WINDOW_WORDS)
        sent = self.sentiment_pipe(window, truncation=True)[0]
        candidates = topics or list(dict.fromkeys(suggest_topics(text) + self.default_topics))
        return ClassificationResult(
            language=detect_language(window),
            sentiment=LabelScore(
                label=str(sent["label"]).lower(), score=round(float(sent["score"]), 4)
            ),
            style=self._rank(window, config.STYLE_LABELS)[0],
            political=self._rank(window, config.POLITICAL_LABELS)[0],
            topics=self._rank(window, candidates, multi_label=True)[:top_k],
        )
