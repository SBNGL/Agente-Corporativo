from pathlib import Path
import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

def dividir_documentos(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documentos)

    return chunks

def obtener_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    
def crear_vectorstore(chunks, modelo_embeddings):

    ruta_faiss = "datos/faiss"

    if os.path.exists(ruta_faiss):
        shutil.rmtree(ruta_faiss)
        print("Índice FAISS anterior descartado. Reconstruyendo...")

    print("Creando vectorstore...")

    vectorstore = FAISS.from_documents(
        chunks,
        modelo_embeddings
    )

    vectorstore.save_local(ruta_faiss)

    print("Vectorstore creado y guardado.")

    return vectorstore


def crear_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "fetch_k": 10
        }
    )