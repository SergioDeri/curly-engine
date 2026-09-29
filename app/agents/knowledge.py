from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from app.rag.retriever import Chunk, Retriever

UNSUPPORTED = "No tengo información sobre eso en las políticas de la tienda."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Respondés consultas de clientes de Tienda Nodo usando exclusivamente los fragmentos "
            "de políticas que siguen. Si los fragmentos no alcanzan para responder, dejá "
            "`sources` vacío y decilo en `answer`. No agregues datos que no estén en el texto.\n\n"
            "{context}\n\n{format_instructions}",
        ),
        ("human", "{question}"),
    ]
)


class GroundedAnswer(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool


class LLMAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(
        description="Nombres de los archivos de política usados, por ejemplo devoluciones.md"
    )


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)


class KnowledgeAgent:
    def __init__(self, retriever: Retriever, llm: BaseChatModel, min_score: float) -> None:
        self._retriever = retriever
        self._min_score = min_score
        parser = PydanticOutputParser(pydantic_object=LLMAnswer)
        chain = PROMPT.partial(format_instructions=parser.get_format_instructions()) | llm | parser
        self._chain = chain.with_retry(
            retry_if_exception_type=(OutputParserException,),
            wait_exponential_jitter=False,
            stop_after_attempt=2,
        ).with_fallbacks(
            [RunnableLambda(lambda _: LLMAnswer(answer=UNSUPPORTED, sources=[]))],
            exceptions_to_handle=(OutputParserException,),
        )

    async def answer(self, question: str) -> GroundedAnswer:
        chunks = [c for c in await self._retriever.search(question) if c.score >= self._min_score]
        if not chunks:
            return GroundedAnswer(answer=UNSUPPORTED, sources=[], grounded=False)

        reply = await self._chain.ainvoke({"question": question, "context": format_context(chunks)})
        retrieved = {c.source for c in chunks}
        cited = sorted({source.split(">")[0].strip() for source in reply.sources})
        if not cited or not set(cited) <= retrieved:
            return GroundedAnswer(answer=UNSUPPORTED, sources=[], grounded=False)
        return GroundedAnswer(answer=reply.answer, sources=cited, grounded=True)
