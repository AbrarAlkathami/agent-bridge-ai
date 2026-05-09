# Agent Bridge AI

A multi-agent AI research assistant that routes user queries to specialized agents — RAG for document Q&A, web search for real-time information, and a research planner for structured research. Built with LangGraph, FastAPI, and Streamlit.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                           │
│            Files Tab  │  RAG Chat Tab  │  Tools Tab  │  Graph Tab  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  HTTP / SSE
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                             │
│                                                                     │
│   POST /chat/        POST /tool-agent/    POST /documents/upload    │
│   POST /search/      GET  /documents/     GET  /documents/{id}/...  │
│   POST /debug/rag    POST /debug/search   GET  /debug/db            │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────┐
              │        Supervisor (LangGraph)          │
              │                                       │
              │  Classifies query into one of:        │
              │    • "rag"   → document intent        │
              │    • "tools" → web / research intent  │
              └────────────┬──────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐      ┌────────────────────────┐
   │    RAG Agent     │      │      Tools Agent        │
   │                  │      │                         │
   │ retrieve_context │      │  web_search (Tavily)    │
   │ tool → DuckDB    │      │  research_planner (LLM) │
   │ similarity search│      │                         │
   └────────┬─────────┘      └──────────┬──────────────┘
            │                           │
            ▼                           ▼
   ┌──────────────────┐      ┌─────────────────────────┐
   │  DuckDB Vector   │      │    Tavily Search API     │
   │  Store           │      │    (real-time web)       │
   │  (text-embedding │      └─────────────────────────┘
   │   -3-large)      │
   └──────────────────┘

         Both agents stream tokens back via SSE → Frontend
```

### Component Map

| Layer | Technology | Role |
|---|---|---|
| Frontend | Streamlit | Chat UI, file management, tool invocation |
| API | FastAPI + Uvicorn | HTTP routing, SSE streaming, request validation |
| Orchestration | LangGraph `StateGraph` | Supervisor node routes query to the right agent |
| RAG Agent | LangChain `create_agent` | Retrieves document chunks then generates an answer |
| Tools Agent | LangChain `create_agent` | Web search and research plan generation |
| Vector Store | DuckDB + LangChain `DuckDB` | Persists and queries embedded document chunks |
| Embeddings | OpenAI `text-embedding-3-large` | Encodes document chunks and queries |
| Web Search | Tavily API | Live internet search |
| LLM | OpenAI GPT-4o (configurable) | Powers all agents and the supervisor |
| Observability | LangSmith (optional) | Traces, evaluation datasets, latency monitoring |

---

## Design Decisions

### 1. LangGraph Supervisor as a Thin Router
The supervisor is intentionally minimal — a single LangGraph node that calls the LLM once with a structured-output schema (`RouteDecision`) and returns either `"rag"` or `"tools"`. It holds no memory between turns. This keeps routing fast, deterministic to the model's classification, and trivially replaceable if the routing logic needs to grow.

### 2. Two Specialized Agents Instead of One General Agent
Rather than one agent with every tool attached, the system splits responsibility:
- **RAG Agent** — always calls `retrieve_context` before answering. Its system prompt forbids skipping retrieval so it never fabricates document content.
- **Tools Agent** — has `web_search` and `research_planner`. The prompt explicitly gates `research_planner` behind explicit user intent so it is never triggered as a side-effect of web search.

This separation makes each agent easier to tune, test, and swap without affecting the other.

### 3. DuckDB as the Vector Store
DuckDB is used as the local vector store instead of a hosted service (e.g. Pinecone, Weaviate). It runs in-process with zero infrastructure — a single `.duckdb` file in `storage/`. This is appropriate for a single-user or small-team deployment and removes a network dependency from the hot path.

### 4. Server-Sent Events (SSE) for Streaming
All chat endpoints return `StreamingResponse` with `text/event-stream`. Each LLM token is flushed as it is generated, giving the frontend a real-time typewriter experience. The event envelope carries a `type` field (`route`, `token`, `chunks`, `[DONE]`) so the frontend can update the UI — showing which agent was selected and the source chunks — without waiting for the full response.

### 5. Document Pipeline (Upload → Chunk → Embed → Store)
Files are saved to `storage/files/{document_id}/` so the original can be previewed or deleted. A temporary copy is created for loading (LangChain loaders require a file path), then deleted after indexing. Chunks carry `document_id`, `source`, and `chunk_index` metadata so the debug endpoints can inspect them and the RAG agent can cite sources.

### 6. Debug Mode Behind a Feature Flag
All `/debug/*` routes check `settings.debug_mode` and return 404 in production. This lets developers inspect raw vector search results and database state during development without exposing an attack surface in production.

### 7. LangSmith Tracing as Opt-In
LangSmith tracing is activated only when both `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY` are present. The project also ships an `eval/` directory with a dataset and evaluation runner, keeping the quality-measurement loop local to the repository.

---

## Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| **DuckDB local vector store** | No infra setup, fast local reads | Does not scale beyond a single process; no replication or concurrent writes across instances |
| **Single-step LLM router** | Simple, fast, no state | One model call is fallible — edge cases between "rag" and "tools" may misroute; no fallback |
| **Two agents with fixed tool sets** | Clean separation, predictable behavior | Adding a third capability (e.g. code execution) requires a new agent and router class |
| **SSE streaming** | Real-time UX, low perceived latency | Cannot use standard `response_model` validation; harder to test; proxies/load-balancers must support chunked transfer |
| **OpenAI-only LLM + Embeddings** | Consistent quality, one API key | Vendor lock-in; switching providers requires changing `model_provider` and re-embedding all stored documents |
| **No conversation memory** | Stateless, horizontally scalable | Each query is independent; multi-turn context is not maintained across requests |
| **File size limit (50 MB)** | Protects server memory during sync processing | Rejects large corpora; async chunking / background jobs would be needed to lift this limit |

---

## Project Structure

```
agent-bridge-ai/
├── agents/
│   ├── graph.py            # LangGraph supervisor (router)
│   ├── model.py            # Model provider singleton
│   ├── rag_agent.py        # RAG agent with retrieve_context tool
│   └── tools_agent.py      # Tools agent with web_search & research_planner
├── core/
│   ├── config.py           # Pydantic settings (env vars)
│   ├── document_repo.py    # DuckDB chunk insert / delete / count
│   └── vector_store.py     # DuckDB vector store singleton
├── embeddings/
│   └── embedding.py        # Embedding model setup
├── eval/
│   ├── dataset.json        # Evaluation Q&A pairs
│   └── run_eval.py         # LangSmith evaluation runner
├── routes/
│   ├── chat_route.py       # POST /chat/ — supervisor + agent SSE stream
│   ├── debug_route.py      # /debug/* — vector search & DB inspection
│   ├── schemas.py          # Shared Pydantic request/response models
│   ├── search.py           # POST /search/ — direct similarity search
│   ├── tool_agent.py       # POST /tool-agent/ — direct tools agent
│   ├── tools.py            # GET /tools/ — list available tools
│   └── upload_file_route.py # POST /documents/uploadfile/
├── services/
│   ├── research_planner_service.py
│   ├── retrieval_service.py
│   ├── upload_service.py
│   └── web_serach_service.py
├── storage/
│   ├── files/              # Original uploaded files (by document_id)
│   └── rag.duckdb          # Embedded document chunks
├── tools/
│   ├── research_planner.py # LangChain tool — structured research plan
│   └── web_search.py       # LangChain tool — Tavily web search
├── frontend.py             # Streamlit UI
├── main.py                 # FastAPI app entry point
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

### Setup

```bash
# Install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
```

**.env** variables:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Powers all LLM calls and embeddings |
| `TAVILY_API_KEY` | Yes | Powers the web search tool |
| `MODEL_NAME` | No | LLM model name (default: `gpt-4o`) |
| `DEBUG_MODE` | No | Enables `/debug/*` routes (default: `false`) |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

### Run

```bash
# Start the API server
uv run uvicorn main:app --reload

# In a separate terminal, start the Streamlit frontend
uv run streamlit run frontend.py
```

API: `http://localhost:8000`  
Frontend: `http://localhost:8501`  
API docs: `http://localhost:8000/docs`
