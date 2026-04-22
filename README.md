# ESG Agent — LLM Agent with RAG + Tools

A lightweight, production-inspired ESG-focused AI agent that retrieves and reasons over sustainability reports using Retrieval-Augmented Generation (RAG) and tool usage.

## Architecture

```
esg_agent/
├── config.py          # Settings, ModelType, ESGCategory enums
├── models.py          # Pydantic data models (ESGEvent, ESGExtractionResult, …)
├── tools.py           # LangChain tools: think_and_write, web_search, calculator, lookup_esg_database
├── rag/
│   └── retriever.py   # In-memory FAISS RAG retriever
├── prompts/
│   └── extraction_prompt.py  # System prompts for extraction and validation
├── extraction.py      # LLM-based ESG event extractor (RAG → LLM → parse)
├── validation.py      # Validation layer (groundedness check)
├── agent.py           # create_esg_agent() factory
├── orchestrator.py    # Top-level pipeline: extract → validate
└── _db.py             # Lightweight in-process session store
```

## Pipeline

```
Documents → RAG Retriever → Context
                          ↓
              LLM Extractor (+ tools)
                          ↓
              ESGExtractionResult (validated=False)
                          ↓
              LLM Validator (groundedness)
                          ↓
              ESGExtractionResult (validated=True/False per event)
```

## Features

| Feature | Description |
|---|---|
| **RAG Pipeline** | FAISS-backed in-memory vector store; chunks documents and retrieves top-k passages |
| **Structured Extraction** | Pydantic-typed `ESGEvent` with company, event, category, confidence, source_excerpt |
| **Tools** | `think_and_write`, `web_search` (Tavily), `calculator`, `lookup_esg_database` |
| **Validation Layer** | LLM cross-checks every event against its cited excerpt; sets `validated` flag |
| **Modular Design** | Each component (retriever, extractor, validator, agent) can be used independently |

## Example Output

```json
{
  "company": "Tesla",
  "event": "announced new manufacturing facility",
  "category": "Expansion",
  "confidence": 0.91,
  "source_excerpt": "Tesla announced a new manufacturing facility in Texas.",
  "validated": true
}
```

## Quick Start

```python
import asyncio
from esg_agent.orchestrator import orchestrate_esg_analysis

result = asyncio.run(
    orchestrate_esg_analysis(
        company="Tesla",
        query="carbon emissions and expansion plans",
        documents=["<full text of Tesla sustainability report>"],
        top_k=5,
    )
)

for event in result.events:
    print(event.model_dump_json(indent=2))
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for OpenAI models |
| `GOOGLE_API_KEY` | — | Required for Gemini models |
| `ESG_DEFAULT_MODEL` | `gpt-4o-mini` | LLM to use |
| `ESG_MAX_WEB_SEARCHES` | `3` | Cap on `web_search` calls per run |
| `ESG_RAG_TOP_K` | `5` | Number of RAG chunks retrieved per query |
| `TAVILY_API_KEY` | — | Required for `web_search` tool |

## Running Tests

```bash
uv run pytest
```
