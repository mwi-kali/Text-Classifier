from pydantic import BaseModel, Field
from typing import List


class ClassificationResult(BaseModel):
    language: str = Field(..., description="Detected language code")
    political: str = Field(..., description="Political leaning")
    sentiment: str = Field(..., description="Positive or negative sentiment")
    style: str = Field(..., description="Formal or informal style")    
    topics: List[str] = Field(..., description="Top predicted topics")


class SummarizationResult(BaseModel):
    summary: str = Field(..., description="Extractive summary text")
