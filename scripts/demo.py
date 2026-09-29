import asyncio
import json
import os
import sys
import uuid

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
# Keeps the demo under the Groq free tier limit of 8000 tokens per minute.
PAUSE = float(os.getenv("DEMO_PAUSE", "20"))

SCENARIOS = {
    "politica": ["¿Cuántos días tengo para devolver un producto y quién paga el envío?"],
    "pedido": ["Hola, ¿en qué estado está mi pedido 1003? Mi email es lucia.fernandez@mail.com"],
    "pedido_y_politica": [
        "Compré unos auriculares in-ear, pedido 1002, email martin.gomez@mail.com. "
        "¿Los puedo devolver? Si no, ¿qué cubre la garantía?"
    ],
    "sin_sustento": ["¿Hacen instalación a domicilio y tienen descuento para estudiantes?"],
    "memoria": [
        "Quiero devolver el pedido 1001",
        "Mi email es lucia.fernandez@mail.com y el motivo es que la pantalla tiene píxeles muertos",
    ],
}


async def chat(client: httpx.AsyncClient, thread_id: str, message: str) -> str:
    answer = []
    payload = {"thread_id": thread_id, "message": message}
    async with client.stream("POST", "/chat", json=payload) as response:
        response.raise_for_status()
        event = None
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:") and event == "token":
                token = json.loads(line.removeprefix("data:"))["token"]
                answer.append(token)
                print(token, end="", flush=True)
            elif line.startswith("data:") and event == "error":
                print(f"[error] {json.loads(line.removeprefix('data:'))['message']}", end="")
    print()
    return "".join(answer)


async def main() -> None:
    first = True
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        for name, messages in SCENARIOS.items():
            thread_id = f"demo-{name}-{uuid.uuid4().hex[:6]}"
            print(f"\n=== {name} ({thread_id})")
            for message in messages:
                if not first:
                    await asyncio.sleep(PAUSE)
                first = False
                print(f"\n> {message}\n")
                await chat(client, thread_id, message)


if __name__ == "__main__":
    asyncio.run(main())
