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
    rag, retriever = crear_rag(llm, retriever)

    print(f"Total de paginas cargadas: {len(documentos)}")
    print(f"Total de fragmentos generados: {len(chunks)}")
    

    print("\n========================================")
    print(" Asistente Inteligente Mercado Central ")
    print("========================================")
    print("Escribe 'salir' para terminar.\n")

    while True:

        pregunta = input("Pregunta: ")

        if pregunta.lower() == "salir":
            print("\n¡Hasta luego!")
            break

       
        respuesta = rag.invoke(pregunta)

        
        documentos_encontrados = retriever.invoke(pregunta)

        print("\nRespuesta:")
        print(respuesta)

        print("\nFuentes:")

        for documento in documentos_encontrados:

            nombre_pdf = Path(documento.metadata["source"]).name
            pagina = documento.metadata["page"] + 1

            print(f"- {nombre_pdf} (Página {pagina})")

        print("-" * 60)



if __name__ == "__main__":
    main()  

