from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def chunk(self, documents: list[Document]) -> list[Document]:
        print(f"[chunker] chunking {len(documents)} pages")
        print(f"[chunker] chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        chunks = self.splitter.split_documents(documents)
        
        print(f"[chunker] produced {len(chunks)} chunks")
        return chunks