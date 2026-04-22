"""RAG retriever: chunks documents and retrieves top-k relevant passages."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class ESGRetriever:
    """Lightweight in-memory RAG retriever backed by FAISS.

    Usage:
        retriever = ESGRetriever()
        retriever.index_documents(["Long sustainability report text..."])
        chunks = retriever.retrieve("carbon emissions target", top_k=5)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._embeddings = OpenAIEmbeddings()
        self._vectorstore: InMemoryVectorStore | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[str]) -> None:
        """Split *documents* into chunks and build a FAISS index.

        Args:
            documents: List of raw text strings representing report pages or
                       sections.  Each string may be arbitrarily long.
        """
        if not documents:
            raise ValueError("At least one document is required for indexing.")

        all_chunks: list[str] = []
        for doc in documents:
            chunks = self._splitter.split_text(doc)
            all_chunks.extend(chunks)

        logger.debug(
            f"Indexed {len(documents)} document(s) → {len(all_chunks)} chunk(s)"
        )
        self._vectorstore = InMemoryVectorStore.from_texts(all_chunks, self._embeddings)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the *top_k* most relevant chunks for *query*.

        Args:
            query: The search query (e.g. a company name + ESG theme).
            top_k: Maximum number of chunks to return.

        Returns:
            List of :class:`RetrievedChunk` objects ordered by relevance (best first).

        Raises:
            RuntimeError: If :meth:`index_documents` has not been called yet.
        """
        if self._vectorstore is None:
            raise RuntimeError(
                "No documents have been indexed. Call index_documents() first."
            )

        results = self._vectorstore.similarity_search_with_score(query, k=top_k)
        chunks = [
            RetrievedChunk(text=doc.page_content, score=float(score), metadata=doc.metadata)
            for doc, score in results
        ]
        logger.debug(f"Retrieved {len(chunks)} chunk(s) for query: {query!r}")
        return chunks

    def retrieve_as_text(self, query: str, top_k: int = 5) -> str:
        """Convenience wrapper that joins retrieved chunks into a single string."""
        chunks = self.retrieve(query, top_k=top_k)
        return "\n\n---\n\n".join(c.text for c in chunks)

    @property
    def is_indexed(self) -> bool:
        return self._vectorstore is not None
