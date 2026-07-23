from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


PROMPT_RAG = """
Eres un asistente virtual especializado en responder consultas sobre la documentación interna de Mercado Central 24h.

Dispones de información proveniente únicamente de los documentos internos de la empresa.

Reglas:

1. Responde únicamente utilizando la información del contexto.

2. No inventes información.

3. Si el contexto contiene información parcial o relacionada, responde con lo que sí está respaldado y menciona claramente si falta detalle específico.

4. Si no aparece ninguna información relevante en el contexto, responde exactamente:

"No encontré información sobre esa consulta en la documentación disponible."

5. Analiza TODO el contexto antes de responder.

6. Si la información está distribuida en varios fragmentos, combínala en una única respuesta completa.

7. No respondas únicamente con el primer fragmento recuperado.

8. Si varios fragmentos contienen información complementaria, intégralos en una sola respuesta coherente y completa.

9. Organiza la respuesta usando listas o viñetas cuando sea posible.

10. Si la consulta es ambigua, solicita únicamente la información necesaria.

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