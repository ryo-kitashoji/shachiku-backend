from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ExcuseRequest(BaseModel):
    question: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ExcuseResponse(BaseModel):
    question: str
    excuse: str
    confidence: float


class ReplySettings(BaseModel):
    userId: str
    channel: str
    replyTo: str


class ReplyMission(BaseModel):
    instruction: str
    goal: str


class ReplyMessage(BaseModel):
    content: str
    timestamp: datetime


class ReplyRequest(BaseModel):
    settings: ReplySettings
    mission: ReplyMission
    message: ReplyMessage


class ReplyResponse(BaseModel):
    reply: str
    replyAt: datetime