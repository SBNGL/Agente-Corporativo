from pathlib import Path
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def cargar_documentos(ruta):

    documentos = []

    for archivo in Path(ruta).glob("*.pdf"):

        try:

            loader = PyMuPDFLoader(str(archivo))
            documentos.extend(loader.load())

            print(f"Archivo cargado: {archivo.name}")

        except Exception as e:

            print(f"Error cargando {archivo.name}: {e}")

    return documentos


def dividir_documentos(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documentos)


def obtener_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def crear_vectorstore(chunks, embeddings):

    ruta_faiss = "datos/faiss"

    if os.path.exists(ruta_faiss):

        print("Cargando índice FAISS...")

        return FAISS.load_local(
            ruta_faiss,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("Creando índice FAISS...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(ruta_faiss)

    print("Índice FAISS creado.")

    return vectorstore


def crear_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_kwargs={
            "k": 6,

        }
    )