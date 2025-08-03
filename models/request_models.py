from pydantic import BaseModel
from typing import Optional


class ExcuseRequest(BaseModel):
    question: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ExcuseResponse(BaseModel):
    question: str
    excuse: str
    confidence: float