from __future__ import annotations

import json
import re
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9']+")
_VOWELS = re.compile(r"[aeiouy]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, candidate: str) -> float:
    ref, cand = _tokens(reference), _tokens(candidate)
    lcs = _lcs(ref, cand)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(cand), lcs / len(ref)
    return 2 * p * r / (p + r)


def _syllables(word: str) -> int:
    groups = _VOWELS.findall(word.lower())
    n = len(groups) - (1 if word.lower().endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    words = _tokens(text)
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return round(206.835 - 1.015 * len(words) / sentences - 84.6 * syl / len(words), 1)


@dataclass
class SummaryEval:
    doc_id: str
    mode: str
    rouge_l: float
    compression: float
    reading_ease: float
    latency_s: float
    peak_mb: float
    forward_passes: int


def evaluate(
    summarizer, corpus: list[dict], modes=("abstractive", "map-reduce")
) -> list[SummaryEval]:
    """``corpus`` items: {"id", "text", "reference"}."""
    rows = []
    for item in corpus:
        for mode in modes:
            fn = summarizer.map_reduce if mode == "map-reduce" else summarizer.abstractive
            calls_before = getattr(summarizer, "calls", 0)
            tracemalloc.start()
            t0 = time.perf_counter()
            summary = fn(item["text"])
            latency = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                SummaryEval(
                    item["id"],
                    mode,
                    round(rouge_l(item["reference"], summary), 4),
                    round(len(summary.split()) / max(1, len(item["text"].split())), 4),
                    flesch_reading_ease(summary),
                    round(latency, 3),
                    round(peak / 1e6, 2),
                    getattr(summarizer, "calls", 0) - calls_before,
                )
            )
    return rows


def load_corpus(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def to_markdown(rows: list[SummaryEval]) -> str:
    cols = list(asdict(rows[0])) if rows else []
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in asdict(r).values()) + " |" for r in rows]
    return "\n".join(lines)
