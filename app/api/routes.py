import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Request
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import AIMessageChunk, HumanMessage
from sse_starlette import EventSourceResponse, ServerSentEvent

from app.api.schemas import (
    ChatRequest,
    HealthResponse,
    HistoryMessage,
    HistoryResponse,
    IngestResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

RATE_LIMITED = "Se alcanzó el límite de consultas al modelo. Probá de nuevo en un minuto."
FAILED = "No pude procesar la consulta. Probá de nuevo en unos minutos."


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def is_rate_limit(error: Exception) -> bool:
    return isinstance(error, ModelRateLimitError) or getattr(error, "status_code", None) == 429


def error_event(message: str) -> ServerSentEvent:
    return ServerSentEvent(json.dumps({"message": message}), event="error")


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/chat", response_class=EventSourceResponse)
async def chat(body: ChatRequest, request: Request) -> EventSourceResponse:
    graph = request.app.state.services.graph

    async def events() -> AsyncIterator[ServerSentEvent]:
        stream = graph.astream(
            {"messages": [HumanMessage(body.message)]},
            thread_config(body.thread_id),
            stream_mode="messages",
        )
        try:
            async for chunk, metadata in stream:
                if metadata.get("langgraph_node") != "respond":
                    continue
                if isinstance(chunk, AIMessageChunk) and chunk.text:
                    yield ServerSentEvent(json.dumps({"token": chunk.text}), event="token")
        except Exception as error:
            if is_rate_limit(error):
                logger.warning("Rate limit del modelo en el thread %s: %s", body.thread_id, error)
                yield error_event(RATE_LIMITED)
            else:
                logger.exception("Falló el grafo en el thread %s", body.thread_id)
                yield error_event(FAILED)
            return
        yield ServerSentEvent(json.dumps({"thread_id": body.thread_id}), event="done")

    return EventSourceResponse(events())


@router.get("/threads/{thread_id}/history")
async def history(
    thread_id: Annotated[str, Path(max_length=64, pattern=r"^[\w-]+$")], request: Request
) -> HistoryResponse:
    state = await request.app.state.services.graph.aget_state(thread_config(thread_id))
    messages = state.values.get("messages", [])
    if not messages:
        raise HTTPException(404, "Conversación inexistente")
    return HistoryResponse(
        thread_id=thread_id,
        messages=[
            HistoryMessage(
                role="cliente" if isinstance(m, HumanMessage) else "asistente", content=m.text
            )
            for m in messages
            if isinstance(m, HumanMessage) or m.name == "assistant"
        ],
    )


@router.post("/ingest")
async def ingest(request: Request) -> IngestResponse:
    ingest_policies = request.app.state.services.ingest
    if ingest_policies is None:
        raise HTTPException(503, "Ingesta no disponible")
    return IngestResponse(chunks=await ingest_policies(reset=True))
