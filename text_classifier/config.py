from __future__ import annotations

import os

SENTIMENT_MODEL = os.getenv("TC_SENTIMENT_MODEL", "tabularisai/multilingual-sentiment-analysis")
ZERO_SHOT_MODEL = os.getenv("TC_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
SUMMARY_MODEL = os.getenv("TC_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")

CLASSIFY_WINDOW_WORDS = 350
SUMMARY_CHUNK_TOKENS = int(os.getenv("TC_SUMMARY_CHUNK_TOKENS", "700"))
SUMMARY_OVERLAP_TOKENS = int(os.getenv("TC_SUMMARY_OVERLAP_TOKENS", "60"))
SUMMARY_BATCH_SIZE = int(os.getenv("TC_SUMMARY_BATCH_SIZE", "4"))
MAX_SUMMARY_ROUNDS = 3
MAX_INPUT_CHARS = 400_000
HTTP_TIMEOUT = 15
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024

DEFAULT_TOPICS = ["technology", "business", "politics", "science", "health", "sports", "culture"]
STYLE_LABELS = ["formal", "informal"]
POLITICAL_LABELS = ["left-leaning", "centrist", "right-leaning"]
SUPPORTED_FILE_EXTENSIONS = {"pdf", "txt", "md", "html", "htm"}
