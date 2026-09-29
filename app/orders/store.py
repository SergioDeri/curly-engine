import asyncio
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import aiofiles

from app.orders.models import Order, ReturnRequest


def parse_table(markdown: str) -> list[dict[str, str]]:
    rows = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return []
    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    return [
        dict(zip(header, (cell.strip() for cell in row.strip("|").split("|")), strict=True))
        for row in rows[2:]
    ]


RETURN_WINDOW = timedelta(days=30)
NON_RETURNABLE_CATEGORIES = {"auriculares-in-ear", "software"}

RejectionCode = Literal[
    "not_found", "not_delivered", "outside_window", "not_returnable", "already_requested"
]

REJECTION_MESSAGES: dict[RejectionCode, str] = {
    "not_found": "No existe un pedido con ese número para ese email.",
    "not_delivered": "El pedido todavía no fue entregado.",
    "outside_window": "Pasaron más de 30 días desde la entrega.",
    "not_returnable": "El producto no admite devolución, solo garantía.",
    "already_requested": "Ya existe una solicitud de devolución para este pedido.",
}


class ReturnRejected(Exception):
    def __init__(self, code: RejectionCode) -> None:
        super().__init__(REJECTION_MESSAGES[code])
        self.code = code


async def read_table(path: Path) -> list[dict[str, str]]:
    async with aiofiles.open(path) as f:
        return parse_table(await f.read())


class OrderStore:
    def __init__(
        self,
        orders_file: Path,
        returns_file: Path,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._orders_file = orders_file
        self._returns_file = returns_file
        self._today = today
        self._write_lock = asyncio.Lock()

    async def find(self, order_id: str, email: str) -> Order | None:
        order_id, email = order_id.strip(), email.strip().lower()
        for row in await read_table(self._orders_file):
            if row["pedido"] == order_id and row["email"].lower() == email:
                return Order.from_row(row)
        return None

    async def get_return(self, order_id: str, email: str) -> ReturnRequest | None:
        order_id, email = order_id.strip(), email.strip().lower()
        for row in await read_table(self._returns_file):
            if row["pedido"] == order_id and row["email"].lower() == email:
                return ReturnRequest.from_row(row)
        return None

    async def request_return(self, order_id: str, email: str, reason: str) -> ReturnRequest:
        order = await self.find(order_id, email)
        if order is None:
            raise ReturnRejected("not_found")
        if order.status != "entregado" or order.delivered_on is None:
            raise ReturnRejected("not_delivered")
        if order.category in NON_RETURNABLE_CATEGORIES:
            raise ReturnRejected("not_returnable")
        if self._today() - order.delivered_on > RETURN_WINDOW:
            raise ReturnRejected("outside_window")

        request = ReturnRequest(
            id=f"D-{order.id}",
            order_id=order.id,
            email=order.email,
            reason=" ".join(reason.replace("|", "/").split()),
            requested_on=self._today(),
        )
        async with self._write_lock:
            if await self.get_return(order.id, order.email):
                raise ReturnRejected("already_requested")
            async with aiofiles.open(self._returns_file, "a") as f:
                await f.write(request.to_row())
        return request
