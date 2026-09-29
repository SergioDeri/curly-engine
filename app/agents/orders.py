from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.orders.models import Order, ReturnRequest
from app.orders.store import REJECTION_MESSAGES, OrderStore, ReturnRejected

PROMPT = (
    "Gestionás pedidos de Tienda Nodo con las herramientas disponibles. Para cualquier "
    "consulta necesitás el número de pedido y el email de la compra; si falta alguno, no "
    "llames a ninguna herramienta e indicá qué dato falta. Si el cliente pregunta si puede "
    "devolver un pedido, consultalo con find_order. Creá una devolución solo si el cliente la "
    "pide y dio el motivo."
)


class OrderLookup(BaseModel):
    order_id: str = Field(description="Número de pedido, por ejemplo 1004")
    email: str = Field(description="Email con el que se hizo la compra")


class ReturnInput(OrderLookup):
    reason: str = Field(min_length=3, description="Motivo de la devolución")


class ToolResult(BaseModel):
    ok: bool
    message: str = ""
    order: Order | None = None
    return_request: ReturnRequest | None = None


def build_order_tools(store: OrderStore) -> list[BaseTool]:
    async def find_order(order_id: str, email: str) -> ToolResult:
        order = await store.find(order_id, email)
        if order is None:
            return ToolResult(ok=False, message=REJECTION_MESSAGES["not_found"])
        return ToolResult(ok=True, order=order)

    async def request_return(order_id: str, email: str, reason: str) -> ToolResult:
        try:
            created = await store.request_return(order_id, email, reason)
        except ReturnRejected as rejected:
            return ToolResult(ok=False, message=str(rejected))
        return ToolResult(ok=True, return_request=created)

    async def get_return(order_id: str, email: str) -> ToolResult:
        existing = await store.get_return(order_id, email)
        if existing is None:
            return ToolResult(ok=False, message="No hay devoluciones para ese pedido y email.")
        return ToolResult(ok=True, return_request=existing)

    def tool(fn, schema: type[BaseModel], description: str) -> BaseTool:
        async def run(**kwargs) -> str:
            return (await fn(**kwargs)).model_dump_json(exclude_none=True)

        return StructuredTool.from_function(
            coroutine=run,
            name=fn.__name__,
            description=description,
            args_schema=schema,
            handle_validation_error=True,
        )

    return [
        tool(find_order, OrderLookup, "Consulta estado, producto y fecha de entrega de un pedido."),
        tool(request_return, ReturnInput, "Crea la solicitud de devolución de un pedido."),
        tool(get_return, OrderLookup, "Consulta si un pedido tiene una devolución solicitada."),
    ]


class OrdersAgent:
    def __init__(self, llm: BaseChatModel, store: OrderStore) -> None:
        tools = build_order_tools(store)
        self._tools = {t.name: t for t in tools}
        self._model = llm.bind_tools(tools)

    async def run(self, task: str) -> str:
        reply = await self._model.ainvoke([SystemMessage(PROMPT), HumanMessage(task)])
        if not reply.tool_calls:
            return reply.text
        results = [await self._call(call) for call in reply.tool_calls]
        return "\n".join(
            f"{call['name']}: {result.content}"
            for call, result in zip(reply.tool_calls, results, strict=True)
        )

    async def _call(self, call: dict) -> ToolMessage:
        tool = self._tools.get(call["name"])
        if tool is None:
            return ToolMessage(f"Herramienta desconocida: {call['name']}", tool_call_id=call["id"])
        return await tool.ainvoke(call)
