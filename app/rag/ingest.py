import asyncio
from pathlib import Path

import aiofiles
from chromadb.api.models.AsyncCollection import AsyncCollection
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "title"), ("##", "section")]

header_splitter = MarkdownHeaderTextSplitter(HEADERS)
size_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def split_policy(source: str, markdown: str) -> list[tuple[str, dict]]:
    chunks = []
    for doc in size_splitter.split_documents(header_splitter.split_text(markdown)):
        # The heading path goes into the text so the embedding knows what the chunk is about.
        heading = " > ".join(doc.metadata[k] for _, k in HEADERS if k in doc.metadata)
        chunks.append((f"{heading}\n\n{doc.page_content}", {"source": source}))
    return chunks


async def ingest_policies(
    collection: AsyncCollection,
    embeddings: Embeddings,
    policies_dir: Path,
    reset: bool = False,
) -> int:
    if reset:
        existing = await collection.get(include=[])
        if existing["ids"]:
            await collection.delete(ids=existing["ids"])

    chunks: list[tuple[str, dict]] = []
    paths = await asyncio.to_thread(lambda: sorted(policies_dir.glob("*.md")))
    for path in paths:
        async with aiofiles.open(path) as f:
            chunks.extend(split_policy(path.name, await f.read()))
    if not chunks:
        return 0

    texts = [text for text, _ in chunks]
    await collection.upsert(
        ids=[f"{meta['source']}:{i}" for i, (_, meta) in enumerate(chunks)],
        documents=texts,
        metadatas=[meta for _, meta in chunks],
        embeddings=await embeddings.aembed_documents(texts),
    )
    return len(chunks)
