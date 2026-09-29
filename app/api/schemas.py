from typing import Literal

from pydantic import BaseModel, Field

ThreadId = Field(min_length=1, max_length=64, pattern=r"^[\w-]+$")


class ChatRequest(BaseModel):
    thread_id: str = ThreadId
    message: str = Field(min_length=1, max_length=2000)


class HistoryMessage(BaseModel):
    role: Literal["cliente", "asistente"]
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class IngestResponse(BaseModel):
    chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
