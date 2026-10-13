from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LabelScore(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)


class ClassificationResult(BaseModel):
    language: str = Field(description="ISO 639-1 code, or 'unknown'")
    sentiment: LabelScore
    style: LabelScore
    political: LabelScore = Field(description="experimental zero-shot political leaning")
    topics: list[LabelScore]


class AnalysisRequest(BaseModel):
    text: str | None = Field(default=None, max_length=400_000)
    url: str | None = None
    classify: bool = True
    summarize: bool = True
    summary_mode: Literal["abstractive", "map-reduce"] = "abstractive"
    topics: list[str] | None = Field(default=None, description="candidate topics to rank")
    # extraction flags (URLs only)
    render_js: bool | None = Field(
        default=None, description="None = auto, True = always use a headless browser"
    )
    include_tables: bool = True
    include_lists: bool = True
    include_code: bool = True
    max_chars: int = Field(default=400_000, ge=1_000, le=400_000, description="extraction depth")

    @model_validator(mode="after")
    def exactly_one_source(self):
        if bool(self.text) == bool(self.url):
            raise ValueError("provide exactly one of 'text' or 'url'")
        return self


class AnalysisResult(BaseModel):
    characters: int
    words: int
    source: dict | None = Field(default=None, description="title, type, headings, pages, tables")
    classification: ClassificationResult | None = None
    summary: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
