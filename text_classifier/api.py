from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from text_classifier import __version__, config, models
from text_classifier.ingestion import IngestionError, bytes_to_text
from text_classifier.schemas import AnalysisRequest, AnalysisResult
from text_classifier.service import analyze

logger = logging.getLogger("text_classifier.api")

app = FastAPI(
    title="Text Classifier & Summarizer",
    version=__version__,
    description="Sentiment, style, topics and summaries for text, URLs, PDFs and YouTube videos.",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "sentiment": config.SENTIMENT_MODEL,
            "zero_shot": config.ZERO_SHOT_MODEL,
            "summary": config.SUMMARY_MODEL,
        },
        "loaded": models.get.cache_info().currsize,
    }


@app.post("/warmup")
async def warmup():
    for load in (models.sentiment, models.zero_shot, models.summarizer):
        await run_in_threadpool(load)
    return {"loaded": models.get.cache_info().currsize}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_endpoint(req: AnalysisRequest):
    try:
        return await run_in_threadpool(analyze, req)
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/analyze/file", response_model=AnalysisResult)
async def analyze_file(file: UploadFile, summarize: bool = True, classify: bool = True):
    data = await file.read()
    try:
        text = bytes_to_text(data, file.filename or "upload.txt")
        req = AnalysisRequest(text=text, summarize=summarize, classify=classify)
        return await run_in_threadpool(analyze, req)
    except (IngestionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
