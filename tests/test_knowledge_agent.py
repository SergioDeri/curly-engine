import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.agents.knowledge import KnowledgeAgent
from app.rag.retriever import Chunk


class StubRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    async def search(self, query: str) -> list[Chunk]:
        return self.chunks


class ExplodingModel(FakeListChatModel):
    responses: list[str] = []

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("the model must not be called")


RETURNS_CHUNK = Chunk(
    text="El cliente puede solicitar la devolución dentro de los 30 días corridos.",
    source="devoluciones.md",
    score=0.82,
)


def llm_reply(answer: str, sources: list[str]) -> str:
    return json.dumps({"answer": answer, "sources": sources})


async def test_answers_without_calling_the_model_when_nothing_relevant_is_found():
    weak = Chunk(text="Envío gratis desde $150.000", source="envios.md", score=0.3)
    agent = KnowledgeAgent(StubRetriever([weak]), ExplodingModel(), min_score=0.6)

    result = await agent.answer("¿Venden consolas usadas?")

    assert not result.grounded
    assert result.sources == []


async def test_answer_cites_the_retrieved_sources_it_used():
    llm = FakeListChatModel(responses=[llm_reply("Tenés 30 días.", ["devoluciones.md"])])
    agent = KnowledgeAgent(StubRetriever([RETURNS_CHUNK]), llm, min_score=0.6)

    result = await agent.answer("¿Cuántos días tengo para devolver?")

    assert result.grounded
    assert result.answer == "Tenés 30 días."
    assert result.sources == ["devoluciones.md"]


async def test_citations_that_include_the_section_count_as_the_file():
    llm = FakeListChatModel(
        responses=[llm_reply("Tenés 30 días.", ["devoluciones.md > Plazo", "devoluciones.md"])]
    )
    agent = KnowledgeAgent(StubRetriever([RETURNS_CHUNK]), llm, min_score=0.6)

    result = await agent.answer("¿Cuántos días tengo para devolver?")

    assert result.grounded
    assert result.sources == ["devoluciones.md"]


async def test_answer_citing_sources_that_were_not_retrieved_is_not_trusted():
    llm = FakeListChatModel(responses=[llm_reply("Tenés 60 días.", ["garantia-extendida.md"])])
    agent = KnowledgeAgent(StubRetriever([RETURNS_CHUNK]), llm, min_score=0.6)

    result = await agent.answer("¿Cuántos días tengo para devolver?")

    assert not result.grounded
    assert "60" not in result.answer


async def test_retries_once_when_the_model_returns_invalid_output():
    llm = FakeListChatModel(
        responses=["Tenés 30 días", llm_reply("Tenés 30 días.", ["devoluciones.md"])]
    )
    agent = KnowledgeAgent(StubRetriever([RETURNS_CHUNK]), llm, min_score=0.6)

    result = await agent.answer("¿Cuántos días tengo para devolver?")

    assert result.grounded
    assert result.sources == ["devoluciones.md"]


async def test_falls_back_to_a_safe_answer_when_the_model_keeps_failing():
    llm = FakeListChatModel(responses=["basura", '{"answer": 1}'])
    agent = KnowledgeAgent(StubRetriever([RETURNS_CHUNK]), llm, min_score=0.6)

    result = await agent.answer("¿Cuántos días tengo para devolver?")

    assert not result.grounded
    assert result.sources == []
