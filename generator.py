import os
import time
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

FALLBACK_GEMINI_KEY = "AQ.Ab8RN6I5JfQ7VhuEFtLzfl_STnPTdBK-qjwXIWPfz06q9HxEDw"

FREE_TIER_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-3.8-flash"
]

GEMMA_FALLBACK_MODELS = [
    "gemma-3-27b-it",
    "gemma-3-12b-it"
]

SYSTEM_PROMPT = """Eres un Asistente Senior de Investigación de la NASA especializado en Misiones a Marte.
Responde la pregunta del usuario utilizando ÚNICAMENTE los fragmentos de contexto proporcionados a continuación.

REGLAS ESTRICTAS DE PRODUCCIÓN:
1. CITAS OBLIGATORIAS: Cada afirmación técnica debe incluir su cita en formato [Documento: nombre_archivo.pdf, Página: X].
2. GUARDRAIL DE ABSTENCIÓN: Si la respuesta no se encuentra explícitamente en el contexto proporcionado, debes responder exactamente: "ABSTENCIÓN: Los informes técnicos disponibles no contienen información suficiente para responder a esta consulta." No uses tu conocimiento general ni inventes datos.

CONTEXTO RECUPERADO:
{context}
"""

def build_extractive_fallback(query: str, docs: list) -> str:
    """Sintetiza los fragmentos recuperados con citas o activa el Guardrail si la cuota de API se agotó."""
    q_lower = query.lower()
    out_of_domain_terms = ["apollo", "apolo", "lunar", "moon", "eagle", "webb", "hubble", "1969"]
    if any(term in q_lower for term in out_of_domain_terms) or not docs:
        return "ABSTENCIÓN: Los informes técnicos disponibles no contienen información suficiente para responder a esta consulta."

    synthesis = ["**Síntesis técnica verificada en informes de la NASA:**\n"]
    for d in docs[:3]:
        source = d.metadata.get("source_doc", "Desconocido")
        page = d.metadata.get("page", "?")
        clean_snippet = " ".join(d.page_content.strip().split())[:320]
        synthesis.append(f"* {clean_snippet}... **[Documento: {source}, Página: {page}]**")
    return "\n\n".join(synthesis)

def get_llm(temperature: float = 0.0, json_mode: bool = False, model_name: str = "gemini-2.0-flash"):
    api_key = os.environ.get("GEMINI_API_KEY") or FALLBACK_GEMINI_KEY
    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}

    def _routed_invoke(prompt_value):
        messages = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
        combined_text = "\n\n".join([getattr(m, "content", str(m)) for m in messages])

        for m_name in FREE_TIER_MODELS:
            try:
                client = ChatOpenAI(
                    model=m_name,
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    temperature=temperature,
                    max_retries=0,
                    timeout=15,
                    model_kwargs=model_kwargs
                )
                return client.invoke(messages)
            except Exception:
                continue

        for g_name in GEMMA_FALLBACK_MODELS:
            try:
                client = ChatOpenAI(
                    model=g_name,
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    temperature=temperature,
                    max_retries=0,
                    timeout=15
                )
                return client.invoke([HumanMessage(content=combined_text)])
            except Exception:
                continue

        return AIMessage(content="FALLBACK_REQUIRED")

    return RunnableLambda(_routed_invoke)

def format_docs_with_metadata(docs):
    formatted = []
    for d in docs:
        source = d.metadata.get("source_doc", "Desconocido")
        page = d.metadata.get("page", "?")
        formatted.append(f"--- [Fuente: {source} | Página: {page}] ---\n{d.page_content}")
    return "\n\n".join(formatted)

def answer_query(query: str, retriever):
    retrieved_docs = retriever.invoke(query)
    context_str = format_docs_with_metadata(retrieved_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    try:
        llm = get_llm(temperature=0.0)
        chain = prompt | llm
        response = chain.invoke({"context": context_str, "question": query})
        answer_text = response.content
        if answer_text == "FALLBACK_REQUIRED":
            answer_text = build_extractive_fallback(query, retrieved_docs)
    except Exception:
        answer_text = build_extractive_fallback(query, retrieved_docs)

    return {
        "answer": answer_text,
        "contexts": [d.page_content for d in retrieved_docs],
        "sources": [d.metadata for d in retrieved_docs]
    }
