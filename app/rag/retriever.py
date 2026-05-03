from app.rag.loader import PDFLoader
from app.rag.chunker import Chunker
from app.rag.embedder import VectorStore


class Retriever:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.store = VectorStore()
        self._indexed = False

    def index(self) -> None:
        collection = self.store.client.get_or_create_collection("docuagent")

        if collection.count() > 0:
            print(f"[retriever] already has {collection.count()} chunks, skipping")
            self._indexed = True
            return

        # only index if empty
        docs = PDFLoader.load_directory(self.data_dir)
        chunker = Chunker()
        chunks = chunker.chunk(docs)
        self.store.store(chunks)
        self._indexed = True
        print("[retriever] indexing complete")

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        if not self._indexed:
            raise RuntimeError("Call index() before retrieve()")
        return self.store.retrieve(query, k)


if __name__ == "__main__":
    retriever = Retriever()
    retriever.index()

    # test 1
    print("\n--- Query 1: salary ---")
    results = retriever.retrieve("What is the expected salary range?")
    for r in results:
        print(f"Page {r['metadata'].get('page')} | {r['content'][:100]}")

    # test 2
    print("\n--- Query 2: LangGraph ---")
    results = retriever.retrieve("What is LangGraph and how does it work?")
    for r in results:
        print(f"Page {r['metadata'].get('page')} | {r['content'][:100]}")

    # test 3 — try indexing again, should skip
    print("\n--- Trying to index again ---")
    retriever.index()
