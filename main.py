from pathlib import Path
from src.llm import obtener_llm
from src.procesamiento_documentos import cargar_documentos, dividir_documentos, obtener_embeddings, crear_vectorstore, crear_retriever
from src.rag import crear_rag


def main():
    llm = obtener_llm()

    documentos = cargar_documentos("datos/pdf")
    chunks = dividir_documentos(documentos)
    modelo_embeddings = obtener_embeddings()
    vectorstore = crear_vectorstore(chunks, modelo_embeddings)
    retriever = crear_retriever(vectorstore)
    rag = crear_rag(llm, retriever)

    print(f"Total de paginas cargadas: {len(documentos)}")
    print(f"Total de fragmentos generados: {len(chunks)}")
    

    print("\n========================================")
    print(" Asistente Inteligente Mercado Central ")
    print("========================================")
    print("Escribe 'salir' para terminar.\n")

    while True:

        pregunta = input("\nPregunta: ")

        if pregunta.lower() == "salir":
            print("\n¡Hasta luego!")
            break

        documentos_encontrados = retriever.invoke(pregunta)

        print("\n===== DOCUMENTOS RECUPERADOS =====")

        for i, documento in enumerate(documentos_encontrados, 1):
            print(f"\nDocumento {i}")
            print("-" * 60)
            print(documento.page_content)

        print("\n==============================")
        
        respuesta = rag.invoke(pregunta)

        print("\nRespuesta:\n")
        print(respuesta)

       

        fuentes = set()

        for documento in documentos_encontrados:

            nombre_pdf = Path(documento.metadata["source"]).name
            pagina = documento.metadata["page"] + 1

            fuentes.add((nombre_pdf, pagina))

        print("\nFuentes:")

        for nombre_pdf, pagina in sorted(fuentes):
            print(f"• {nombre_pdf} (Página {pagina})")

        print("\n" + "-" * 70)



if __name__ == "__main__":
    main()  

