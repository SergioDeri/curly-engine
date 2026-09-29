import json

import httpx
import pytest
from langchain_core.exceptions import ModelRateLimitError

from app.agents.graph import build_graph
from app.api.main import Services, create_app
from tests.conftest import StubKnowledge, plan, scripted


@pytest.fixture
async def client(checkpointer):
    llm = scripted(plan(), "Hola, ¿en qué\nte ayudo?")
    graph = build_graph(llm, StubKnowledge(), store=None, checkpointer=checkpointer)
    app = create_app(Services(graph=graph))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def read_events(response: httpx.Response) -> list[tuple[str, dict]]:
    events, name = [], "message"
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line.removeprefix("data:"))))
            name = "message"
    return events


async def test_chat_streams_only_the_final_answer(client):
    async with client.stream("POST", "/chat", json={"thread_id": "abc", "message": "Hola"}) as r:
        events = await read_events(r)

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    tokens = "".join(data["token"] for name, data in events if name == "token")
    assert tokens == "Hola, ¿en qué\nte ayudo?"
    assert events[-1] == ("done", {"thread_id": "abc"})


class ProviderRateLimit(Exception):
    status_code = 429


class FailingGraph:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def astream(self, *args, **kwargs):
        raise self._error
        yield


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ModelRateLimitError("429"), "Se alcanzó el límite de consultas al modelo"),
        (ProviderRateLimit("429"), "Se alcanzó el límite de consultas al modelo"),
        (RuntimeError("boom"), "No pude procesar la consulta"),
    ],
)
async def test_chat_reports_model_failures_as_an_error_event(error, message):
    app = create_app(Services(graph=FailingGraph(error)))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/chat", json={"thread_id": "e", "message": "Hola"}) as r:
            events = await read_events(r)

    name, data = events[-1]
    assert name == "error"
    assert data["message"].startswith(message)


@pytest.mark.parametrize(
    "payload",
    [
        {"thread_id": "abc", "message": ""},
        {"thread_id": "con espacios", "message": "Hola"},
        {"message": "Hola"},
    ],
)
async def test_chat_rejects_invalid_requests(client, payload):
    response = await client.post("/chat", json=payload)

    assert response.status_code == 422


async def test_history_returns_only_customer_and_assistant_messages(client):
    async with client.stream("POST", "/chat", json={"thread_id": "h1", "message": "Hola"}) as r:
        await read_events(r)

    response = await client.get("/threads/h1/history")

    assert response.json() == {
        "thread_id": "h1",
        "messages": [
            {"role": "cliente", "content": "Hola"},
            {"role": "asistente", "content": "Hola, ¿en qué\nte ayudo?"},
        ],
    }


async def test_history_of_unknown_thread_is_not_found(client):
    assert (await client.get("/threads/nadie/history")).status_code == 404


async def test_health(client):
    response = await client.get("/health")

    assert response.json() == {"status": "ok"}
