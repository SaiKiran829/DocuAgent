import shutil
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.agent.graph import AgentGraph
from app.rag.retriever import Retriever
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] starting up...")
    retriver = Retriever(data_dir=DATA_DIR)
    retriver.index()
    print("[api] ready")
    yield


app = FastAPI(title="DocuAgent API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request / Response Models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    k: int = 4
    thread_id: str = "default"  # default for backwards compatibility
    
class SourceModel(BaseModel):
    source: str
    page: int
    
class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    has_context: bool
    intent: str

class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int
    
# ─── Startup ────────────────────────────────────────────────────────────

DATA_DIR = "data"
    
# ─── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "DocuAgent API is healthy"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # save uploaded file to data directory
    save_path = Path(DATA_DIR) / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[api] saved file to {save_path}")
    
    # re-index with new file
    # delete old chroma db first to avoid duplicates
    
    
        
    retriever = Retriever(data_dir=DATA_DIR)
    retriever.store.clear()
    retriever.index()
    
    #get chunk count
    collection = retriever.store.client.get_collection("docuagent")
    chunk_count = collection.count()
    
    return IngestResponse(
        message="File ingested and indexed successfully",
        filename=file.filename,
        chunks_stored=chunk_count
    )
    
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    agent = AgentGraph.build()
    
    # pass thread_id in config — this is how LangGraph finds conversation history
    config = {"configurable": {"thread_id": request.thread_id}}
    
    result = await run_in_threadpool(agent.invoke, {
        "messages": [HumanMessage(content=request.query)],
    }, config=config)
    
    
    # extract sources from chunks
    sources = []
    
    for chunk in result.get("chunks", []):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", 0)
        
        # avoid duplicates
        source_model = SourceModel(source=source, page=page)
        if source_model not in sources:
            sources.append(source_model)
        
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    
    return AskResponse(
        answer=result.get("answer", "Sorry, I don't know the answer to that."),
        sources=sources,
        has_context=result.get("has_enough_context", False),
        intent=result.get("intent", "unknown")
    )
    

    
