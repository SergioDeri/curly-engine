from datetime import date

import pytest
from langchain_core.messages import HumanMessage

from app.agents.graph import build_graph
from app.orders.store import OrderStore
from tests.conftest import StubKnowledge, plan, scripted, tool_call


@pytest.fixture
def store(tmp_path):
    orders = tmp_path / "orders.md"
    returns = tmp_path / "returns.md"
    orders.write_text(
        "| pedido | email | producto | categoria | monto | estado | fecha_entrega |\n"
        "|---|---|---|---|---|---|---|\n"
        "| 1001 | lucia@mail.com | Notebook Lenovo | notebooks | 899999 | entregado | 2026-09-12 |\n"
    )
    returns.write_text(
        "| devolucion | pedido | email | motivo | fecha_solicitud |\n|---|---|---|---|---|\n"
    )
    return OrderStore(orders, returns, today=lambda: date(2026, 9, 24))


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def test_policy_question_is_answered_by_the_knowledge_agent(checkpointer, store):
    knowledge = StubKnowledge()
    llm = scripted(
        plan(("knowledge", "Plazo para devolver un producto")),
        "Tenés 30 días desde la entrega para devolverlo.",
    )
    graph = build_graph(llm, knowledge, store=store, checkpointer=checkpointer)

    state = await graph.ainvoke(
        {"messages": [HumanMessage("¿Cuántos días tengo para devolver?")]}, config("t1")
    )

    assert knowledge.questions == ["Plazo para devolver un producto"]
    assert state["messages"][-1].content == "Tenés 30 días desde la entrega para devolverlo."


async def test_orders_agent_creates_a_return_with_a_single_model_call(checkpointer, store):
    llm = scripted(
        plan(("orders", "Devolver pedido 1001, email lucia@mail.com, motivo: no enciende")),
        tool_call("request_return", order_id="1001", email="lucia@mail.com", reason="No enciende"),
        "Listo, tu devolución D-1001 quedó registrada.",
    )
    graph = build_graph(llm, StubKnowledge(), store=store, checkpointer=checkpointer)

    state = await graph.ainvoke(
        {"messages": [HumanMessage("Quiero devolver el 1001, lucia@mail.com, no enciende")]},
        config("t2"),
    )

    saved = await store.get_return("1001", "lucia@mail.com")
    report = next(m for m in state["messages"] if m.name == "orders")
    assert saved is not None and saved.reason == "No enciende"
    assert '"id":"D-1001"' in report.content
    assert state["messages"][-1].content == "Listo, tu devolución D-1001 quedó registrada."


async def test_orders_agent_asks_for_missing_data_without_tools(checkpointer, store):
    llm = scripted(
        plan(("orders", "Consultar el pedido 1001")),
        "Falta el email de la compra.",
        "¿Me pasás el email con el que hiciste la compra?",
    )
    graph = build_graph(llm, StubKnowledge(), store=store, checkpointer=checkpointer)

    state = await graph.ainvoke({"messages": [HumanMessage("¿Y mi pedido 1001?")]}, config("t3"))

    report = next(m for m in state["messages"] if m.name == "orders")
    assert report.content == "Falta el email de la compra."
    assert state["messages"][-1].content == "¿Me pasás el email con el que hiciste la compra?"


async def test_supervisor_plans_both_agents_in_one_call(checkpointer, store):
    knowledge = StubKnowledge("La garantía cubre fallas de fábrica.", ["garantia.md"])
    llm = scripted(
        plan(
            ("orders", "Consultar pedido 1001, email lucia@mail.com"),
            ("knowledge", "Qué cubre la garantía"),
        ),
        tool_call("find_order", order_id="1001", email="lucia@mail.com"),
        "Tu notebook está entregada y la garantía cubre fallas de fábrica.",
    )
    graph = build_graph(llm, knowledge, store=store, checkpointer=checkpointer)

    state = await graph.ainvoke(
        {"messages": [HumanMessage("Pedido 1001, lucia@mail.com. ¿Qué cubre la garantía?")]},
        config("t4"),
    )

    assert [m.name for m in state["messages"]] == [None, "orders", "knowledge", "assistant"]
    assert knowledge.questions == ["Qué cubre la garantía"]


async def test_conversation_history_is_kept_per_thread(checkpointer, store):
    llm = scripted(plan(), "¡Hola Lucía!", plan(), "Te llamás Lucía.")
    graph = build_graph(llm, StubKnowledge(), store=store, checkpointer=checkpointer)

    await graph.ainvoke({"messages": [HumanMessage("Hola, soy Lucía")]}, config("lucia"))
    await graph.ainvoke({"messages": [HumanMessage("¿Cómo me llamo?")]}, config("lucia"))

    history = (await graph.aget_state(config("lucia"))).values["messages"]
    other = await graph.aget_state(config("otro"))
    assert [m.content for m in history] == [
        "Hola, soy Lucía",
        "¡Hola Lucía!",
        "¿Cómo me llamo?",
        "Te llamás Lucía.",
    ]
    assert "messages" not in other.values


async def test_plan_is_capped_at_two_steps(checkpointer, store):
    knowledge = StubKnowledge()
    llm = scripted(plan(*[("knowledge", "Plazos")] * 4), "Tenés 30 días.")
    graph = build_graph(llm, knowledge, store=store, checkpointer=checkpointer)

    state = await graph.ainvoke({"messages": [HumanMessage("¿Plazos?")]}, config("cap"))

    assert len(knowledge.questions) == 2
    assert state["messages"][-1].content == "Tenés 30 días."
