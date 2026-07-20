from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


PROMPT_RAG = """
Eres un asistente virtual especializado en responder consultas sobre la documentación interna de Mercado Central 24h.

Dispones de información proveniente únicamente de los documentos internos de la empresa.

Reglas:

1. Responde únicamente utilizando la información del contexto.

2. No inventes información.

3. Si la respuesta no aparece en el contexto responde exactamente:

"No encontré información sobre esa consulta en la documentación disponible."

4. Analiza TODO el contexto antes de responder.

5. Si la información está distribuida en varios fragmentos, combínala en una única respuesta completa.

6. No respondas únicamente con el primer fragmento recuperado.

7. Si varios fragmentos contienen información complementaria, intégralos en una sola respuesta coherente y completa.

8. Organiza la respuesta usando listas o viñetas cuando sea posible.

9. Si la consulta es ambigua, solicita únicamente la información necesaria.

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_RAG)


def formatear_documentos(documentos):
    if not documentos:
        return "No se encontró información relevante."

    return "\n\n".join(
        documento.page_content
        for documento in documentos
    )


def crear_rag(llm, retriever):

    return (
        {
            "context": retriever | RunnableLambda(formatear_documentos),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )