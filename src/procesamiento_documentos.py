from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def cargar_documentos(ruta):
    documentos = []

    for n in Path(ruta).glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            documentos.extend(loader.load())
            print(f"Archivo cargado: {n.name}")
        except Exception as e:
            print(f"Error cargando archivo: {n.name}: {e}")

    return documentos

def  dividir_documentos(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documentos)

    return chunks

def obtener_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    
def crear_vectorstore(chunks, modelo_embeddings):
    vectorstore = FAISS.from_documents(
        chunks,
        modelo_embeddings
    )
    return vectorstore

def crear_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)
    return retriever