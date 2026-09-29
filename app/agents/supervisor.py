from typing import Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Coordinás el soporte al cliente de Tienda Nodo, una tienda online de electrónica. "
            "Leé la conversación y armá el plan para responder el último mensaje del cliente, "
            "con como máximo un paso por agente:\n"
            "- knowledge: preguntas sobre políticas de envíos, devoluciones, garantía, medios de "
            "pago o preguntas frecuentes.\n"
            "- orders: consultar un pedido puntual o solicitar una devolución. Incluí en la tarea "
            "el número de pedido, el email y el motivo de devolución que el cliente haya dado en "
            "cualquier mensaje de la conversación.\n\n"
            "Cada tarea es la consulta que el agente tiene que resolver, nunca la respuesta: no "
            "supongas datos de las políticas ni de los pedidos.\n\n"
            "Dejá el plan vacío si el mensaje es un saludo, está fuera de tema, falta un dato del "
            "cliente o la conversación ya tiene la información para contestar.\n\n"
            "{format_instructions}",
        ),
        MessagesPlaceholder("messages"),
        ("human", "Armá el plan."),
    ]
)


class Step(BaseModel):
    agent: Literal["knowledge", "orders"]
    task: str = Field(description="Consulta autocontenida para el agente")


class Plan(BaseModel):
    steps: list[Step] = Field(default_factory=list)


def build_supervisor(llm: BaseChatModel) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=Plan)
    chain = PROMPT.partial(format_instructions=parser.get_format_instructions()) | llm | parser
    return chain.with_fallbacks(
        [RunnableLambda(lambda _: Plan())],
        exceptions_to_handle=(OutputParserException,),
    )
