import os
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def extract_and_chunk_pdfs(pdf_folder: str):
    """Extrae texto de los PDFs con metadatos de archivo y página, y genera fragmentos."""
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder, exist_ok=True)
        return []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return []

    raw_docs = []
    for filename in sorted(pdf_files):
        file_path = os.path.join(pdf_folder, filename)
        try:
            with fitz.open(file_path) as pdf_doc:
                for page_index, page in enumerate(pdf_doc):
                    text = page.get_text("text").strip()
                    if text:
                        raw_docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source_doc": filename,
                                    "page": page_index + 1
                                }
                            )
                        )
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

    if not raw_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(raw_docs)


def build_vector_store(chunks, persist_dir: str = "./chroma_db"):
    """Construye y persiste la base vectorial ChromaDB validando que existan fragmentos."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if not chunks:
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore
