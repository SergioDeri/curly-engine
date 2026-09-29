from datetime import date

import pytest

from app.orders.store import OrderStore, ReturnRejected

ORDERS = """# Pedidos

| pedido | email | producto | categoria | monto | estado | fecha_entrega |
|---|---|---|---|---|---|---|
| 1001 | lucia@mail.com | Notebook Lenovo | notebooks | 899999 | entregado | 2026-09-12 |
| 1002 | lucia@mail.com | Auriculares in-ear Sony | auriculares-in-ear | 129999 | entregado | 2026-09-18 |
| 1003 | lucia@mail.com | Mouse Logitech | perifericos | 119999 | en camino | |
| 1004 | lucia@mail.com | Monitor Samsung | monitores | 459999 | entregado | 2026-08-24 |
| 1005 | lucia@mail.com | Teclado Redragon | perifericos | 64999 | entregado | 2026-08-25 |
| 1006 | lucia@mail.com | Licencia Microsoft 365 | software | 89999 | entregado | 2026-09-20 |
"""

RETURNS = """# Devoluciones

| devolucion | pedido | email | motivo | fecha_solicitud |
|---|---|---|---|---|
"""


@pytest.fixture
def store(tmp_path):
    orders = tmp_path / "orders.md"
    returns = tmp_path / "returns.md"
    orders.write_text(ORDERS)
    returns.write_text(RETURNS)
    return OrderStore(orders, returns, today=lambda: date(2026, 9, 24))


async def test_finds_order_when_number_and_email_match(store):
    order = await store.find("1001", "lucia@mail.com")

    assert order is not None
    assert order.product == "Notebook Lenovo"
    assert order.status == "entregado"
    assert order.delivered_on == date(2026, 9, 12)


async def test_email_match_ignores_case_and_surrounding_spaces(store):
    assert await store.find("1001", "  Lucia@Mail.com ") is not None


async def test_does_not_reveal_order_to_a_different_email(store):
    assert await store.find("1001", "otra@mail.com") is None


async def test_return_request_is_persisted(store, tmp_path):
    created = await store.request_return("1001", "lucia@mail.com", "No me convenció")

    reopened = OrderStore(tmp_path / "orders.md", tmp_path / "returns.md")
    saved = await reopened.get_return("1001", "lucia@mail.com")

    assert saved == created
    assert saved.reason == "No me convenció"
    assert saved.requested_on == date(2026, 9, 24)


async def test_return_is_allowed_on_the_last_day_of_the_window(store):
    assert await store.request_return("1005", "lucia@mail.com", "Falla una tecla")


@pytest.mark.parametrize(
    ("order_id", "email", "code"),
    [
        ("1004", "lucia@mail.com", "outside_window"),
        ("1003", "lucia@mail.com", "not_delivered"),
        ("1002", "lucia@mail.com", "not_returnable"),
        ("1006", "lucia@mail.com", "not_returnable"),
        ("1001", "otra@mail.com", "not_found"),
        ("9999", "lucia@mail.com", "not_found"),
    ],
)
async def test_return_is_rejected_when_policy_does_not_allow_it(store, order_id, email, code):
    with pytest.raises(ReturnRejected) as rejected:
        await store.request_return(order_id, email, "Motivo")

    assert rejected.value.code == code
    assert await store.get_return(order_id, email) is None


async def test_only_one_return_per_order(store):
    await store.request_return("1001", "lucia@mail.com", "No me convenció")

    with pytest.raises(ReturnRejected) as rejected:
        await store.request_return("1001", "lucia@mail.com", "Otra vez")

    assert rejected.value.code == "already_requested"
