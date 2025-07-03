import logging
import re
import spacy
import string
import subprocess
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect, LangDetectException
from text_classifier.config import CLASSIFY_WINDOW, MAX_WORKERS
from threading import Lock
from transformers import pipeline, Pipeline
from typing import List, Dict, Optional


class Classifier:
    def __init__(self, default_topics: Optional[List[str]] = None):
        self._lock = Lock()
        self.device = 0 if torch.cuda.is_available() else -1
        self.sentiment_pipe: Pipeline = pipeline(
            "sentiment-analysis",
            model="tabularisai/multilingual-sentiment-analysis",
            device=self.device,
        )
        self.zero_shot_pipe: Pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device,
        )
        
        try:
            self._nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            logging.info("Downloading spaCy 'xx_ent_wiki_sm' model...")
            subprocess.run([
                "python", "-m", "spacy", "download", "xx_ent_wiki_sm"
            ], check=True)
            self._nlp = spacy.load("xx_ent_wiki_sm")
            
        self.default_topics = default_topics or ["technology", "politics", "sports"]

    @staticmethod
    def _truncate(text: str) -> str:
        return text[:CLASSIFY_WINDOW]

    def _detect_language(self, text: str) -> str:
        txt = text.strip()
        if not txt:
            return "unknown"
        try:
            return detect(txt)
        except LangDetectException:
            return "unknown"

    def _suggest_topics(self, text: str, top_k: int) -> List[str]:
        doc = self._nlp(text)
        freq: Dict[str, int] = {}
        
        for ent in doc.ents:
            raw = ent.text.strip().lower()
            if not raw or raw.isdigit():
                continue
            
            if all(char in string.punctuation or char.isspace() for char in raw):
                continue
            
            words = raw.split()
            if len(words) < 1 or len(words) > 3:
                continue
            
            cleaned = " ".join(
                tok.text for tok in self._nlp(raw)
                if not tok.is_stop and tok.is_alpha
            )
            if not cleaned:
                continue
            freq[cleaned] = freq.get(cleaned, 0) + 1
        if not freq:
            return self.default_topics
        sorted_topics = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:top_k]]

    def _zero_shot_classify(
        self, text: str, labels: List[str], multi_label: bool = False
    ) -> List[str]:
        try:
            with self._lock:
                res = self.zero_shot_pipe(
                    text, candidate_labels=labels, multi_label=multi_label
                )
            return res.get('labels', [])
        except Exception as e:
            logging.error(f"Zero-shot classification error: {e}")
            return []

    def _sentiment_classify(self, text: str) -> str:
        try:
            with self._lock:
                pred = self.sentiment_pipe(text)[0]
            return pred.get('label', 'unknown').lower()
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return 'unknown'

    def classify_all(
        self,
        text: str,
        topics: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, object]:
        truncated = self._truncate(text)
        language = self._detect_language(text)

        if topics is None:
            topics = self._suggest_topics(text, top_k)

        if not topics:
            topics = self.default_topics
            
        tasks = {
            'sentiment': lambda: self._sentiment_classify(truncated),
            'style': lambda: (
                self._zero_shot_classify(truncated, ['formal', 'informal']) or ['unknown']
            )[0],
            'political': lambda: (
                self._zero_shot_classify(truncated, ['left', 'center', 'right']) or ['unknown']
            )[0],
            'topics': lambda: self._zero_shot_classify(truncated, topics)[:top_k],
        }
        
        results: Dict[str, object] = {'language': language}
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(fn): name for name, fn in tasks.items()}
            for fut in as_completed(future_map):
                name = future_map[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:
                    logging.error(f"Error in {name}: {e}")
                    results[name] = [] if name == 'topics' else 'unknown'
        return results
