from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from functools import cache

from text_classifier import config

logger = logging.getLogger(__name__)
_lock = threading.Lock()

PipelineFactory = Callable[[str, str], Callable]


def _hf_pipeline(task: str, model: str):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info("loading %s (%s) on %s", model, task, "cuda" if device == 0 else "cpu")
    return pipeline(task, model=model, device=device)


@cache
def get(task: str, model: str):
    with _lock:
        return _hf_pipeline(task, model)


def sentiment():
    return get("sentiment-analysis", config.SENTIMENT_MODEL)


def zero_shot():
    return get("zero-shot-classification", config.ZERO_SHOT_MODEL)


def summarizer():
    return get("summarization", config.SUMMARY_MODEL)
