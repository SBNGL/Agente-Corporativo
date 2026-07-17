import streamlit as st
from pathlib import Path

from src.llm import obtener_llm
from src.procesamiento_documentos import (
    cargar_documentos,
    dividir_documentos,
    obtener_embeddings,
    crear_vectorstore,
    crear_retriever
)
from src.rag import crear_rag


st.set_page_config(
    page_title="Mercado Central 24h",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Asistente Inteligente - Mercado Central 24h")

st.markdown("""
Realiza preguntas sobre:

- Horarios
- Proveedores
- Atención al cliente
- Reglamento interno
- Procedimientos operativos
- Preguntas frecuentes
""")


@st.cache_resource
def inicializar_rag():

    llm = obtener_llm()

    documentos = cargar_documentos("datos/pdf")

    chunks = dividir_documentos(documentos)

    embeddings = obtener_embeddings()

    vectorstore = crear_vectorstore(
        chunks,
        embeddings
    )

    retriever = crear_retriever(vectorstore)

    rag = crear_rag(
        llm,
        retriever
    )

    return rag, retriever


try:
    rag, retriever = inicializar_rag()
except Exception as e:
    st.error(f"Error al inicializar la aplicación:\n\n{e}")
    raise


# Historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []


# Mostrar historial
for mensaje in st.session_state.messages:

    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])


pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": pregunta
        }
    )

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.spinner("Consultando la documentación..."):

        respuesta = rag.invoke(pregunta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta
        }
    )

    with st.chat_message("assistant"):
        st.markdown(respuesta)

    documentos = retriever.invoke(pregunta)

    fuentes = set()

    for documento in documentos:

        nombre = Path(documento.metadata["source"]).name
        pagina = documento.metadata["page"] + 1

        fuentes.add((nombre, pagina))

    st.divider()

    st.subheader("📄 Fuentes consultadas")

    for nombre, pagina in sorted(fuentes):
        st.write(f"• **{nombre}** - Página {pagina}")