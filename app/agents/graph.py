from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.knowledge import GroundedAnswer
from app.agents.orders import OrdersAgent
from app.agents.supervisor import build_supervisor
from app.orders.store import OrderStore

RESPONDER_PROMPT = (
    "Sos el asistente de soporte de Tienda Nodo. Escribí la respuesta final al cliente en "
    "español rioplatense, breve y cordial. Usá solo la información que aportaron los agentes "
    "en esta conversación, sin agregar datos, pasos, equivalencias ni aclaraciones propias. "
    "Si un agente indicó que no tiene la información, decile al cliente que no la tenés, sin "
    "pedirle otros datos. Para consultar un pedido hacen falta el número de pedido y el email "
    "de la compra, y para devolverlo también el motivo: si falta alguno, pedilo sin dejar de "
    "responder lo demás. Si la consulta no tiene relación con la tienda, explicá amablemente "
    "en qué podés ayudar."
)

MAX_STEPS = 2


class Knowledge(Protocol):
    async def answer(self, question: str) -> GroundedAnswer: ...


class State(MessagesState):
    steps: list[dict[str, str]]


def build_graph(
    llm: BaseChatModel,
    knowledge: Knowledge,
    store: OrderStore,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    supervisor_chain = build_supervisor(llm)
    orders = OrdersAgent(llm, store)

    async def supervisor(state: State) -> dict:
        if not isinstance(state["messages"][-1], HumanMessage):
            return {"steps": state["steps"][1:]}
        plan = await supervisor_chain.ainvoke({"messages": state["messages"]})
        return {"steps": [step.model_dump() for step in plan.steps[:MAX_STEPS]]}

    async def knowledge_agent(state: State) -> dict:
        result = await knowledge.answer(state["steps"][0]["task"])
        sources = ", ".join(result.sources) or "ninguna"
        report = f"{result.answer}\n\nFuentes: {sources}"
        return {"messages": [AIMessage(report, name="knowledge")]}

    async def orders_agent(state: State) -> dict:
        report = await orders.run(state["steps"][0]["task"])
        return {"messages": [AIMessage(report, name="orders")]}

    async def respond(state: State) -> dict:
        reply = await llm.ainvoke(
            [
                SystemMessage(RESPONDER_PROMPT),
                *state["messages"],
                HumanMessage("Escribí la respuesta final al cliente."),
            ]
        )
        return {"messages": [AIMessage(reply.text, name="assistant")]}

    def next_step(state: State) -> Literal["knowledge", "orders", "respond"]:
        return state["steps"][0]["agent"] if state["steps"] else "respond"

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("knowledge", knowledge_agent)
    graph.add_node("orders", orders_agent)
    graph.add_node("respond", respond)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", next_step)
    graph.add_edge("knowledge", "supervisor")
    graph.add_edge("orders", "supervisor")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
