import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.knowledge import GroundedAnswer


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*replies: str | AIMessage) -> ScriptedModel:
    messages = [r if isinstance(r, AIMessage) else AIMessage(r) for r in replies]
    return ScriptedModel(messages=iter(messages))


def plan(*steps: tuple[str, str]) -> str:
    return json.dumps({"steps": [{"agent": agent, "task": task} for agent, task in steps]})


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


class StubKnowledge:
    def __init__(self, answer: str = "Tenés 30 días.", sources: list[str] | None = None) -> None:
        self.questions: list[str] = []
        self._answer = GroundedAnswer(
            answer=answer, sources=sources or ["devoluciones.md"], grounded=True
        )

    async def answer(self, question: str) -> GroundedAnswer:
        self.questions.append(question)
        return self._answer


@pytest.fixture
async def checkpointer(tmp_path):
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        yield saver
