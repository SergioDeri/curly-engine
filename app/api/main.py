from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from langgraph.graph.state import CompiledStateGraph

from app.api.routes import router


@dataclass
class Services:
    graph: CompiledStateGraph
    ingest: Callable[..., Awaitable[int]] | None = None


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            yield
            return

        from app.core.bootstrap import build_services

        async with build_services() as built:
            app.state.services = built
            yield

    app = FastAPI(title="Tienda Nodo - Asistente de soporte", lifespan=lifespan)
    app.include_router(router)
    if services is not None:
        app.state.services = services
    return app
