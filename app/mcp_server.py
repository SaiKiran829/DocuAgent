from mcp.server.fastmcp import FastMCP
from app.rag.retriever import Retriever
from app.rag.loader import PDFLoader
from pathlib import Path


# FastMCP is like FastAPI but for tools
# Instead of HTTP endpoints, you define tools
mcp = FastMCP("DocuAgent Tools")

retriever = Retriever()
retriever.index()

# ─── Tool 1 ───────────────────────────────────────────────────────────────────
# @mcp.tool() is like @app.get() in FastAPI
# It exposes this function as a callable tool
# Any agent can call it by name: "search_documents"

@mcp.tool()
def search_document(query:str, k:int=4) -> list[dict]:
    """Search the vector store for relevant document chunks."""
    print(f"[mcp] search_documents called: '{query}'")
    results = retriever.retrieve(query, k=k)
    return results
    
    
# ─── Tool 2 ───────────────────────────────────────────────────────────────────

@mcp.tool()
def get_document_list() -> list[str]:
    """List all PDF files that have been ingested."""
    print(f"[mcp] get_document_list called")
    data_dir = Path("data")
    pdfs = [file_path.name for file_path in data_dir.rglob("*") if file_path.is_file() and file_path.suffix.lower() == ".pdf"]
    return pdfs
    
    
# ─── Tool 3 ───────────────────────────────────────────────────────────────────

@mcp.tool()
def get_page_content(filename:str, page:int) -> str:
    """Get the raw text content of a specific page from a PDF."""
    print(f"[mcp] get_page_content called: {filename}, page {page}")

    file_path = next(
        (
            candidate
            for candidate in Path("data").rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".pdf" and candidate.name == filename
        ),
        None,
    )
    if file_path is None:
        return f"File not found: {filename}"

    docs = PDFLoader.load(str(file_path))

    for doc in docs:
        if doc.metadata.get("page") == page:
            return doc.page_content
    
    return f"Page {page} not found in {filename}"
    
if __name__ == "__main__":
    print("[mcp] starting DocuAgent MCP server...")
    mcp.run()