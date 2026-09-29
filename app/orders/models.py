from datetime import date
from typing import Literal

from pydantic import BaseModel

OrderStatus = Literal["en preparacion", "en camino", "entregado", "cancelado"]


class Order(BaseModel):
    id: str
    email: str
    product: str
    category: str
    amount: int
    status: OrderStatus
    delivered_on: date | None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Order":
        return cls(
            id=row["pedido"],
            email=row["email"],
            product=row["producto"],
            category=row["categoria"],
            amount=int(row["monto"]),
            status=row["estado"],
            delivered_on=row["fecha_entrega"] or None,
        )


class ReturnRequest(BaseModel):
    id: str
    order_id: str
    email: str
    reason: str
    requested_on: date

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "ReturnRequest":
        return cls(
            id=row["devolucion"],
            order_id=row["pedido"],
            email=row["email"],
            reason=row["motivo"],
            requested_on=row["fecha_solicitud"],
        )

    def to_row(self) -> str:
        cells = (self.id, self.order_id, self.email, self.reason, self.requested_on)
        return "| " + " | ".join(map(str, cells)) + " |\n"
