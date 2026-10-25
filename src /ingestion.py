import os
import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def extract_and_chunk_pdfs(pdf_folder: str) -> list[Document]:
    raw_docs = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            doc = fitz.open(filepath)
            
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text").strip()
                if len(text) > 100:  # Ignorar páginas vacías o solo con imágenes
                    raw_docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source_doc": filename,
                                "page": page_num + 1
                            }
                        )
                    )
                    
    # Chunking: fragmentos de ~800 caracteres con 150 de solapamiento
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(raw_docs)

def build_vector_store(chunks: list[Document], persist_dir: str = "./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore
