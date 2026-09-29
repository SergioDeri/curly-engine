from typing import Protocol

from chromadb.api.models.AsyncCollection import AsyncCollection
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class Chunk(BaseModel):
    text: str
    source: str
    score: float


class Retriever(Protocol):
    async def search(self, query: str) -> list[Chunk]: ...


class PolicyRetriever:
    def __init__(self, collection: AsyncCollection, embeddings: Embeddings, k: int) -> None:
        self._collection = collection
        self._embeddings = embeddings
        self._k = k

    async def search(self, query: str) -> list[Chunk]:
        vector = await self._embeddings.aembed_query(query)
        result = await self._collection.query(query_embeddings=[vector], n_results=self._k)
        return [
            Chunk(text=text, source=meta["source"], score=1 - distance)
            for text, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
