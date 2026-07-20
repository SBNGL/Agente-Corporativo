from pathlib import Path
import os
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
        chunk_size=1500,
        chunk_overlap=250,
    )

    chunks = splitter.split_documents(documentos)

    return chunks

def obtener_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    
def crear_vectorstore(chunks, modelo_embeddings):

    ruta_faiss = "datos/faiss"

    if os.path.exists(ruta_faiss):

        print("Cargando índice FAISS...")

        vectorstore = FAISS.load_local(
            ruta_faiss,
            modelo_embeddings,
            allow_dangerous_deserialization=True
        )

    else:

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
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 12,
            "lambda_mult": 0.7
        }
    )