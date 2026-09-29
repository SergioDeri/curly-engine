import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

import chromadb
from chromadb.api import AsyncClientAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agents.graph import build_graph
from app.agents.knowledge import KnowledgeAgent
from app.api.main import Services
from app.core.config import get_settings
from app.core.llm import build_chat_model, build_embeddings
from app.core.tracing import setup_tracing
from app.orders.store import OrderStore
from app.rag.ingest import ingest_policies
from app.rag.retriever import PolicyRetriever


async def connect_chroma(host: str, port: int, attempts: int = 10) -> AsyncClientAPI:
    for _ in range(attempts - 1):
        try:
            return await chromadb.AsyncHttpClient(host=host, port=port)
        except Exception:
            await asyncio.sleep(2)
    return await chromadb.AsyncHttpClient(host=host, port=port)


@asynccontextmanager
async def build_services() -> AsyncIterator[Services]:
    settings = get_settings()
    setup_tracing(settings)

    llm = build_chat_model(settings)
    embeddings = build_embeddings(settings)

    chroma = await connect_chroma(settings.chroma_host, settings.chroma_port)
    collection = await chroma.get_or_create_collection(
        settings.chroma_collection,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )
    ingest = partial(ingest_policies, collection, embeddings, settings.policies_dir)
    if await collection.count() == 0:
        await ingest()

    knowledge = KnowledgeAgent(
        PolicyRetriever(collection, embeddings, k=settings.retrieval_k),
        llm,
        min_score=settings.retrieval_min_score,
    )
    store = OrderStore(settings.orders_file, settings.returns_file)

    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as checkpointer:
        graph = build_graph(llm, knowledge, store, checkpointer)
        yield Services(graph=graph, ingest=ingest)
