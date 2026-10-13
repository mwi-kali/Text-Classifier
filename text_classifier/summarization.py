from __future__ import annotations

import logging
import re
from collections.abc import Callable

from text_classifier import config, models

logger = logging.getLogger(__name__)
_SENT = re.compile(r"(?<=[.!?])\s+")


def tidy_summary(text: str) -> str:
    """Remove the space CNN/DailyMail-trained models put before punctuation ("it ." -> "it.")."""
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


class Summarizer:
    def __init__(
        self,
        pipe: Callable | None = None,
        chunk_tokens: int = config.SUMMARY_CHUNK_TOKENS,
        overlap_tokens: int | None = None,
        batch_size: int = config.SUMMARY_BATCH_SIZE,
    ):
        if overlap_tokens is None:  # default overlap, scaled down for small chunks
            overlap_tokens = min(config.SUMMARY_OVERLAP_TOKENS, chunk_tokens // 4)
        if not 0 <= overlap_tokens < chunk_tokens:
            raise ValueError("need 0 <= overlap_tokens < chunk_tokens")
        self._pipe = pipe
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.batch_size = batch_size
        self.calls = 0  # forward passes, for benchmarking

    @property
    def pipe(self):
        return self._pipe if self._pipe is not None else models.summarizer()

    def _count(self, text: str) -> int:
        tok = getattr(self.pipe, "tokenizer", None)
        return len(tok(text, add_special_tokens=False)["input_ids"]) if tok else len(text.split())

    def _units(self, text: str) -> list[tuple[str, int]]:
        """Sentences with token counts; very long sentences are split into word windows."""
        out = []
        for sent in _SENT.split(text.strip()):
            if not sent:
                continue
            n = self._count(sent)
            if n <= self.chunk_tokens:
                out.append((sent, n))
                continue
            words = sent.split()
            step = max(1, int(len(words) * self.chunk_tokens / n) - 1)
            for i in range(0, len(words), step):
                piece = " ".join(words[i : i + step])
                out.append((piece, self._count(piece)))
        return out

    def chunk(self, text: str) -> list[str]:
        """
        Pack sentences into chunks of at most ``chunk_tokens``, each beginning with up to
        ``overlap_tokens`` worth of trailing sentences from the previous chunk.
        """
        units = self._units(text)
        chunks: list[str] = []
        i = 0
        while i < len(units):
            size, j = 0, i
            while j < len(units) and (size + units[j][1] <= self.chunk_tokens or j == i):
                size += units[j][1]
                j += 1
            chunks.append(" ".join(u[0] for u in units[i:j]))
            if j >= len(units):
                break
            k, back = j, 0
            while k - 1 > i and back + units[k - 1][1] <= self.overlap_tokens:
                k -= 1
                back += units[k][1]
            i = k
        return chunks

    def _summarize_chunks(self, chunks: list[str]) -> list[str]:
        """Summarise in batches, preserving order. Length bounds scale with each batch's input."""
        out: list[str] = []
        for b in range(0, len(chunks), self.batch_size):
            batch = chunks[b : b + self.batch_size]
            n = min(self._count(c) for c in batch)
            max_len = max(30, min(200, int(n * 0.6)))
            min_len = min(max_len - 5, max(10, int(n * 0.15)))
            res = self.pipe(
                batch, max_length=max_len, min_length=min_len, do_sample=False, truncation=True
            )
            self.calls += 1
            # a pipeline returns one dict per input (or a nested list when given a list)
            out.extend(
                tidy_summary((r[0] if isinstance(r, list) else r)["summary_text"]) for r in res
            )
        return out

    def abstractive(self, text: str) -> str:
        """Summarise each chunk; join the chunk summaries in document order."""
        text = text.strip()
        if len(text.split()) < 40:
            return text  # too short to be worth summarising
        return " ".join(self._summarize_chunks(self.chunk(text)))

    def _pack(self, partials: list[str]) -> list[str]:
        groups, cur, size = [], [], 0
        for p in partials:
            n = self._count(p)
            if cur and size + n > self.chunk_tokens:
                groups.append(" ".join(cur))
                cur, size = [], 0
            cur.append(p)
            size += n
        if cur:
            groups.append(" ".join(cur))
        return groups

    def map_reduce(self, text: str) -> str:
        """
        Map: summarise every chunk. Reduce: pack partials into model-sized groups and
        summarise each group, level by level, until one summary remains (tree reduce).
        """
        text = text.strip()
        if len(text.split()) < 40:
            return text
        partials = self._summarize_chunks(self.chunk(text))
        for _ in range(config.MAX_SUMMARY_ROUNDS):
            if len(partials) == 1:
                break
            partials = self._summarize_chunks(self._pack(partials))
        return " ".join(partials)
