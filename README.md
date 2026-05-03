# DocuAgent

An enterprise-grade document Q&A agent built with LangGraph, RAG, and AWS Bedrock. Upload any PDF and ask questions — the agent retrieves relevant context, answers with source citations, and knows when it doesn't have enough information.

---

## Demo

> Upload a PDF → Ask a question → Get a grounded answer with citations

📹 [Watch Demo on Loom](#) ← replace with your Loom link

---

## Architecture

```
User (React + TypeScript)
        ↓
FastAPI REST API (async, Pydantic)
        ↓
LangGraph Agent (StateGraph)
  ├── extract_query_node   → extracts query from messages into state
  ├── classify_node        → classifies intent: question / acknowledge / joke
  ├── retrieve_node        → semantic search via ChromaDB
  ├── answer_node          → grounded answer with citations (AWS Bedrock)
  ├── acknowledge_node     → empathetic response to statements
  ├── joke_node            → humor responses
  └── error_node           → graceful fallback after 3 retry attempts
        ↓
ChromaDB (persistent vector store)
        ↓
AWS Bedrock — Claude Haiku (or Anthropic API directly)
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Agent Orchestration | LangGraph | Stateful multi-agent graphs with conditional routing and checkpointing |
| RAG Pipeline | ChromaDB + RecursiveCharacterTextSplitter | Local vector store, semantic chunking that preserves meaning |
| LLM | AWS Bedrock (Claude Haiku) | Enterprise multi-model support within AWS ecosystem |
| API | FastAPI + Pydantic | Async, auto-documented, type-safe REST API |
| Tool Layer | MCP Server | Decoupled tools — any agent can call search_documents without knowing the implementation |
| Containerisation | Docker | Eliminates environment inconsistency |
| Orchestration | Kubernetes + Helm | Self-healing, load balanced, secrets management |
| Frontend | React + TypeScript + Vite | Fast, type-safe UI with PDF upload and chat interface |

---

## Key Features

- **Multi-node LangGraph agent** — explicit state machine with conditional routing, not a black box
- **RAG with citations** — every answer includes source filename and page number
- **Hallucination guard** — uses L2 distance threshold to detect low-relevance retrieval and returns "I don't have enough information" instead of hallucinating
- **Retry logic** — 3 automatic retries on LLM failure using tuple-return pattern (not exceptions), routes to error node on exhaustion
- **MCP server** — exposes `search_documents`, `get_document_list`, `get_page_content` as reusable tools
- **AWS Bedrock integration** — runtime provider switching between Bedrock and Anthropic without code changes
- **Production Kubernetes deployment** — 2 replicas, K8s Secrets for credentials, self-healing pods

---

## Project Structure

```
docuagent/
├── app/
│   ├── agent/
│   │   ├── state.py         # AgentState TypedDict — all state fields defined here
│   │   ├── nodes.py         # All node functions + _invoke_with_retry helper
│   │   ├── router.py        # Routing logic — route_by_intent, route_after_action
│   │   └── graph.py         # LangGraph StateGraph compilation — singleton pattern
│   ├── rag/
│   │   ├── loader.py        # PyPDFLoader — single file and directory loading
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter — configurable chunk size
│   │   ├── embedder.py      # ChromaDB VectorStore — store and retrieve
│   │   └── retriever.py     # End-to-end retrieval — index once, query many times
│   ├── core/
│   │   └── config.py        # LLMClient singleton — Bedrock + Anthropic, runtime switching
│   ├── api/
│   │   └── routes.py        # FastAPI — /health, /ingest, /ask
│   └── mcp_server.py        # MCP tool server — FastMCP with 3 tools
├── k8s/
│   ├── deployment.yaml      # 2 replicas, envFrom secretRef
│   └── service.yaml         # NodePort service
├── data/                    # PDF storage + ChromaDB persistence
├── Dockerfile               # Multi-stage, dependency-cached build
├── .env.example             # Required environment variables
└── main.py                  # Uvicorn entry point
```

---

## RAG Pipeline — How It Works

```
1. INGEST
   PDF file
     → PyPDFLoader (page-by-page loading with metadata)
     → RecursiveCharacterTextSplitter
          chunk_size=1000, overlap=200
          separators: ["\n\n", "\n", ".", " ", ""]
          tries paragraph splits first, falls back to sentence, word, character
     → ChromaDB (persistent, L2 distance)

2. RETRIEVE
   User query
     → ChromaDB similarity search (top k=4 chunks)
     → Distance threshold filter (< 1.5 = relevant)
     → If no relevant chunks → has_enough_context = False

3. ANSWER
   Retrieved chunks + query
     → Prompt: "Answer using ONLY this context. Cite source and page."
     → AWS Bedrock (Claude Haiku)
     → Answer with citations
```

---

## LangGraph Agent — How It Works

```
State: AgentState (TypedDict)
  messages: Annotated[List[BaseMessage], operator.add]  ← accumulates
  chunks:   Annotated[list, operator.add]               ← accumulates
  query:    str                                          ← single writer
  intent:   str                                          ← single writer
  answer:   str                                          ← single writer
  has_enough_context: bool
  error:    Optional[str]
  retry_count: int

Flow:
  START
    → extract_query   (messages[-1].content → state.query)
    → classify        (LLM classifies: question / acknowledge / joke)
    → [conditional]
        question   → retrieve → answer → [conditional] → END or error
        acknowledge → acknowledge → [conditional] → END or error
        joke        → joke → [conditional] → END or error
        error       → error → END
```

---

## Error Handling Strategy

Every node uses a `_invoke_with_retry` helper that:
1. Tries the LLM call up to 3 times
2. Returns a `tuple[str, str | None]` — `(response, error)` — instead of raising exceptions
3. Keeps the node in control — it can update `retry_count` and `error` in state
4. Router detects `error` in state and routes to `error_node`
5. `error_node` returns a graceful fallback message — user never sees a crash

**Why tuple returns instead of exceptions?**
Exceptions crash the node and LangGraph loses control of state. Tuple returns let the node decide what to do with failures — retry, fallback, or escalate.

---

## MCP Server

The MCP server exposes the RAG retrieval layer as reusable tools:

```python
@mcp.tool()
def search_documents(query: str, k: int = 4) -> list[dict]:
    """Search the vector store for relevant document chunks."""

@mcp.tool()
def get_document_list() -> list[str]:
    """List all PDF files that have been ingested."""

@mcp.tool()
def get_page_content(filename: str, page: int) -> str:
    """Get the raw text content of a specific page from a PDF."""
```

Any external agent can call these tools without knowing the ChromaDB implementation. This decouples the retrieval layer from the agent layer — multiple agents across different projects can reuse the same tool server.

---

## How to Run Locally

**Prerequisites:** Python 3.11+, uv, Node 20+

**1. Clone and set up:**
```bash
git clone https://github.com/Saikiran829/docuagent
cd docuagent
uv sync
cp .env.example .env
# add your API keys to .env
```

**2. Run the backend:**
```bash
uv run python main.py
```
API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

**3. Run the frontend:**
```bash
cd docuagent-ui
npm install
npm run dev
```
UI available at `http://localhost:5173`

**4. Run the MCP server (optional):**
```bash
uv run python -m app.mcp_server
```

---

## How to Deploy on Kubernetes

```bash
# Step 1 — point to minikube's Docker
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Step 2 — build image inside minikube
docker build -t docuagent:latest .

# Step 3 — create secrets from .env
kubectl create secret generic docuagent-secrets --from-env-file=.env

# Step 4 — deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Step 5 — get URL
minikube service docuagent-service --url
```

**Verify deployment:**
```bash
kubectl get pods          # should show 2 Running pods
kubectl logs -l app=docuagent  # check logs
```

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```
ANTHROPIC_KEY=your_anthropic_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-south-1
```

---



## What's Next (v2 Roadmap)

- [ ] RAGAS evaluation — faithfulness, context precision, answer relevancy scoring
- [ ] Streaming responses — FastAPI StreamingResponse + React EventSource
- [ ] Multi-document comparison — answer questions across multiple PDFs simultaneously
- [ ] Conversation memory — persist chat history across sessions using LangGraph checkpointing
- [ ] AWS EKS deployment — move from minikube to production Kubernetes on AWS

---

## Built By

Saikiran Pasapula — Software Engineer, Full Stack & Agentic AI  
[LinkedIn](https://www.linkedin.com/in/sai-kiran-4a4761243) | [GitHub](https://github.com/Saikiran829)