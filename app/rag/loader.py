from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class PDFLoader:
    
    @staticmethod
    def load(file_path: str) -> list[Document]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        print(f"[loader] loading: {path.name}")
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        print(f"[loader] loaded {len(pages)} pages from {path.name}")
        return pages
    
    @staticmethod
    def load_directory(directory_path: str) -> list[Document]:
        path = Path(directory_path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        pdf_files = sorted(
            file_path
            for file_path in path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() == ".pdf"
        )
        
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in directory: {directory_path}")

        all_documents = []
        for pdf_file in pdf_files:
            docs = PDFLoader.load(str(pdf_file))
            all_documents.extend(docs)

        return all_documents

        