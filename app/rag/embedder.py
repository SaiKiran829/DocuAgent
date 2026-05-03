from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
import chromadb

class VectorStore:
    
    _instance = None
    
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "docuagent"
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def store(self, chunks: list[Document]) -> None:
        print(f"[vectorstore] storing {len(chunks)} chunks...")
        
        collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[vectorstore] stored successfully")
        
    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        print(f"[vectorstore] retrieving for: '{query}'")
        
        collection= self.client.get_collection(name=self.collection_name)
        
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        chunks = []
        
        for i in range(len(results["documents"][0])):
            chunks.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
            
        print(f"[vectorstore] found {len(chunks)} relevant chunks")
        return chunks
    
    def clear(self) -> None:
        print("[vectorstore] clearing collection...")
        self.client.delete_collection(self.collection_name)
        print("[vectorstore] cleared")

if __name__ == "__main__":
    from app.rag.loader import PDFLoader
    from app.rag.chunker import Chunker

    # load and chunk
    docs = PDFLoader.load_directory("data")
    chunker = Chunker()
    chunks = chunker.chunk(docs)

    # store
    store = VectorStore()
    store.store(chunks)

    # retrieve
    results = store.retrieve("What is the expected salary range?", k=3)

    print("\n--- Results ---")
    for i, result in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(f"Content: {result['content'][:200]}")
        print(f"Source: {result['metadata'].get('source', 'unknown')}")
        print(f"Page: {result['metadata'].get('page', 'unknown')}")
        print(f"Distance: {result['distance']:.4f}")