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

# Configuración de página
st.set_page_config(
    page_title="Mercado Central 24h",
    page_icon="🛒",
    layout="wide"
)


custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp, input, button, select, textarea {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background: radial-gradient(circle at 50% 50%, #0f172a 0%, #070a13 100%) !important;
    color: #e2e8f0 !important;
}

div[data-testid="stSidebar"] {
    background-color: rgba(10, 15, 30, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.title-container {
    padding: 2.5rem 1.5rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 24px;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(10px);
}

.suggestions-panel {
    padding: 1.4rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.42) 0%, rgba(15, 23, 42, 0.72) 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(10px);
    height: 100%;
}

.suggestion-chip {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    margin: 0.3rem 0.35rem 0.3rem 0;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #cbd5e1;
    font-size: 0.88rem;
}

.gradient-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem;
    letter-spacing: -0.5px;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.15rem;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.5;
}

div[data-testid="stChatInput"] {
    border-radius: 30px !important;
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    backdrop-filter: blur(15px) !important;
    box-shadow: 0 -8px 30px rgba(0, 0, 0, 0.5) !important;
    padding: 8px !important;
    margin-top: 1rem !important;
    margin-bottom: 10px !important;
    transition: border-color 0.3s ease;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99, 102, 241, 0.6) !important;
}

div[data-testid="stChatInput"] textarea {
    color: #f1f5f9 !important;
    background-color: transparent !important;
    border: none !important;
    font-size: 1.05rem !important;
}

div[data-testid="stChatMessage"] {
    border-radius: 20px !important;
    padding: 1.25rem !important;
    margin-bottom: 1.25rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stChatMessage"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stChatMessage"][data-testid$="user"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(59, 130, 246, 0.12) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
}

div[data-testid="stChatMessage"][data-testid$="assistant"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.6) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.stButton > button {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    color: #f1f5f9 !important;
    text-align: left !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    height: 100% !important;
    min-height: 110px !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

.stButton > button:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(30, 41, 59, 0.7) 100%) !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.2) !important;
}

div[data-testid="stExpander"] {
    background-color: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    margin-top: 1rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
}

/* Sidebar Styling elements */
.sidebar-logo {
    text-align: center;
    margin-bottom: 2rem;
}

.sidebar-title {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 0.5rem;
}

.stat-box {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    text-align: center;
}

.stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #818cf8;
}

.stat-lbl {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stSidebar"] button {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 12px !important;
    color: #fca5a5 !important;
    padding: 0.6rem !important;
    min-height: 0px !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stSidebar"] button:hover {
    background: rgba(239, 68, 68, 0.2) !important;
    border-color: rgba(239, 68, 68, 0.6) !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
    transform: translateY(-2px) !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)

def inicializar_rag():
    llm = obtener_llm()
    documentos = cargar_documentos("datos/PDF")
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
    
    # Generar estadísticas
    unique_docs = len(set(Path(doc.metadata["source"]).name for doc in documentos))
    pages_count = len(documentos)
    chunks_count = len(chunks)
    
    stats = {
        "docs_count": unique_docs,
        "pages_count": pages_count,
        "chunks_count": chunks_count
    }
    
    return rag, retriever, stats


try:
    rag, retriever, stats = inicializar_rag()
except Exception as e:
    st.error(f"Error al inicializar la aplicación:\n\n{e}")
    raise


# Inicializar variables de estado
if "messages" not in st.session_state:
    st.session_state.messages = []


# Panel lateral (Sidebar)
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span style="font-size: 3.5rem;">🛒</span>
        <div class="sidebar-title">Mercado Central 24h</div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.2rem;">Asistente Inteligente RAG</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 1.5rem 0;' />", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size:1.05rem; font-weight:600; color:#e2e8f0; margin-bottom: 0.8rem;'>📊 Estado de la Base RAG</h3>", unsafe_allow_html=True)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{stats['docs_count']}</div>
            <div class="stat-lbl">PDFs</div>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-val">{stats['pages_count']}</div>
            <div class="stat-lbl">Páginas</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="stat-box" style="margin-top: -0.5rem;">
        <div class="stat-val" style="color: #a855f7;">{stats['chunks_count']}</div>
        <div class="stat-lbl">Fragmentos de Texto</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 1.5rem 0;' />", unsafe_allow_html=True)
    
    # Botón para borrar historial
    if st.button("🧹 Limpiar Historial de Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Cabecera principal o mini-cabecera según el estado del chat
if not st.session_state.messages:
    col_title, col_suggestions = st.columns([1.7, 1.0])

    with col_title:
        st.markdown("""
        <div class="title-container">
            <div class="gradient-title">🛒 Asistente Inteligente</div>
            <div class="subtitle">Pregúntame sobre horarios, políticas de proveedores, atención al cliente, reglamento interno y procedimientos operativos de Mercado Central 24h.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_suggestions:
        st.markdown("""
        <div class="suggestions-panel">
            <div style="font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.9rem;">💡 Preguntas que puedes hacer</div>
            <div class="suggestion-chip">¿Cuáles son los horarios de atención?</div>
            <div class="suggestion-chip">¿Qué políticas de devoluciones hay?</div>
            <div class="suggestion-chip">¿Cuáles son los requisitos de entrega?</div>
            <div class="suggestion-chip">¿Qué normas de seguridad aplican?</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

else:
    # Mini Header when chat has started
    st.markdown("""
    <div style="padding: 0.8rem 1.5rem; margin-bottom: 1.5rem; background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
        <span style="font-size: 1.25rem; font-weight: 800; background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🛒 Mercado Central 24h</span>
        <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 500;">Asistente Activo</span>
    </div>
    """, unsafe_allow_html=True)


# Mostrar historial de mensajes
for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])
        # Mostrar fuentes si existen y es el asistente
        if mensaje["role"] == "assistant" and "sources" in mensaje and mensaje["sources"]:
            with st.expander("🔍 Ver Fuentes Consultadas", expanded=False):
                for nombre, pagina in mensaje["sources"]:
                    st.markdown(f"• **{nombre}** — Página {pagina}")


# Recibir entrada del usuario
pregunta = None
with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input("Escribe tu pregunta...", key="chat_text_input")
    submitted = st.form_submit_button("Enviar")

if submitted and user_question:
    pregunta = user_question.strip()


# Procesar nueva pregunta si existe
if pregunta:
    # 1. Agregar pregunta del usuario
    st.session_state.messages.append(
        {
            "role": "user",
            "content": pregunta
        }
    )

    with st.chat_message("user"):
        st.markdown(pregunta)
        

    with st.spinner("Consultando la documentación del supermercado..."):

        try:

            respuesta = rag.invoke(pregunta)

            documentos_encontrados = retriever.invoke(pregunta)

            fuentes = set()

            for doc in documentos_encontrados:

                nombre = Path(doc.metadata["source"]).name
                pagina = doc.metadata["page"] + 1

                fuentes.add((nombre, pagina))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": respuesta,
                    "sources": sorted(fuentes)
                }
            )

            st.rerun()

        except Exception as e:

            st.error(f"Error al procesar la consulta: {e}")