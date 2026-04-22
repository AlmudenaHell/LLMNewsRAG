"""Unit tests for esg_agent.rag.retriever."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from esg_agent.rag.retriever import ESGRetriever, RetrievedChunk


def _make_retriever_with_mock_vectorstore(chunks: list[str]) -> ESGRetriever:
    """Return an ESGRetriever whose FAISS vectorstore is mocked."""
    retriever = ESGRetriever.__new__(ESGRetriever)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    retriever._splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64
    )
    retriever._embeddings = MagicMock()

    mock_store = MagicMock()
    mock_docs = [
        (MagicMock(page_content=c, metadata={}), float(i + 1))
        for i, c in enumerate(chunks)
    ]
    mock_store.similarity_search_with_score.return_value = mock_docs
    retriever._vectorstore = mock_store

    return retriever


class TestESGRetriever:
    def test_is_indexed_false_before_indexing(self):
        with patch("esg_agent.rag.retriever.OpenAIEmbeddings"):
            retriever = ESGRetriever()
        assert retriever.is_indexed is False

    def test_retrieve_raises_when_not_indexed(self):
        with patch("esg_agent.rag.retriever.OpenAIEmbeddings"):
            retriever = ESGRetriever()
        with pytest.raises(RuntimeError, match="No documents have been indexed"):
            retriever.retrieve("some query")

    def test_retrieve_returns_chunks(self):
        chunks = ["Tesla reduced emissions.", "Tesla expanded solar capacity."]
        retriever = _make_retriever_with_mock_vectorstore(chunks)

        results = retriever.retrieve("Tesla emissions", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].text == "Tesla reduced emissions."

    def test_retrieve_as_text_joins_chunks(self):
        chunks = ["chunk one", "chunk two"]
        retriever = _make_retriever_with_mock_vectorstore(chunks)

        text = retriever.retrieve_as_text("query", top_k=2)

        assert "chunk one" in text
        assert "chunk two" in text
        assert "---" in text  # separator between chunks

    def test_index_documents_raises_on_empty_list(self):
        with patch("esg_agent.rag.retriever.OpenAIEmbeddings"):
            retriever = ESGRetriever()
        with pytest.raises(ValueError, match="At least one document"):
            retriever.index_documents([])

    @patch("esg_agent.rag.retriever.InMemoryVectorStore")
    @patch("esg_agent.rag.retriever.OpenAIEmbeddings")
    def test_index_documents_builds_vectorstore(self, mock_embeddings, mock_store):
        retriever = ESGRetriever(chunk_size=100, chunk_overlap=10)
        retriever.index_documents(["Short document about ESG reporting."])

        assert mock_store.from_texts.called
        assert retriever.is_indexed
