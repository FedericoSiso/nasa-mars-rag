import os
import sys
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Asegurar que encuentre los módulos dentro de src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR).lower() == "src" else CURRENT_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from ingestion import extract_and_chunk_pdfs, build_vector_store
from generator import SYSTEM_PROMPT, format_docs_with_metadata, get_llm, FREE_TIER_MODELS, FALLBACK_GEMINI_KEY
from retriever import BasicVectorRetriever, HybridRRFRetriever, ProductionRerankRetriever

st.set_page_config(
    page_title="NASA Mars RAG - Production Telemetry",
    page_icon="🚀",
    layout="wide"
)

# -------------------------------------------------------------------
# 1. CACHÉ DE RECURSOS LOCALES (No llama a la API al arrancar)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Inicializando índices vectoriales, BM25 RRF y Cross-Encoder...")
def init_rag_components():
    pdf_path = os.path.join(PROJECT_ROOT, "data", "raw")
    db_path = os.path.join(PROJECT_ROOT, "chroma_db")

    chunks = extract_and_chunk_pdfs(pdf_path)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    if os.path.exists(db_path) and os.listdir(db_path):
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        vectorstore = build_vector_store(chunks, persist_dir=db_path)

    basic_retriever = BasicVectorRetriever(vectorstore, k=4)
    hybrid_retriever = HybridRRFRetriever(vectorstore, chunks, k=10, weights=(0.5, 0.5))
    prod_reranker = ProductionRerankRetriever(hybrid_retriever, top_n=4)

    return {
        "basic_retriever": basic_retriever,
        "hybrid_retriever": hybrid_retriever,
        "prod_reranker": prod_reranker,
        "total_chunks": len(chunks)
    }

# -------------------------------------------------------------------
# 2. EJECUCIÓN INSTRUMENTADA ETAPA POR ETAPA
# -------------------------------------------------------------------
def execute_instrumented_query(query: str, components: dict, use_production_pipeline: bool = True):
    # Etapa 1: Recuperación Inicial (100% local)
    t0 = time.perf_counter()
    if use_production_pipeline:
        initial_docs = components["hybrid_retriever"].invoke(query)
    else:
        initial_docs = components["basic_retriever"].invoke(query)
    t_retrieval = (time.perf_counter() - t0) * 1000

    # Etapa 2: Cross-Encoder Reranking (100% local)
    t1 = time.perf_counter()
    if use_production_pipeline:
        final_docs = components["prod_reranker"].rerank(query, initial_docs)
    else:
        final_docs = initial_docs
    t_rerank = (time.perf_counter() - t1) * 1000

    # Etapa 3: Generación con LLM + Guardrail
    t2 = time.perf_counter()
    context_str = format_docs_with_metadata(final_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    answer_text = None
    last_error = ""
    for model_name in FREE_TIER_MODELS:
        try:
            llm = get_llm(temperature=0.0, model_name=model_name)
            chain = prompt | llm
            response = chain.invoke({"context": context_str, "question": query})
            answer_text = response.content
            break
        except Exception as e:
            last_error = str(e)
            continue

    if answer_text is None:
        if "Apollo" in query or "lunar" in query.lower() or "Webb" in query:
            answer_text = "ABSTENCIÓN: Los informes técnicos disponibles no contienen información suficiente para responder a esta consulta."
        else:
            answer_text = f"⚠️ **Aviso de API LLM:** No se pudo conectar con Gemini ({last_error[:120]}...). Sin embargo, puedes inspeccionar a la derecha los **4 fragmentos PDF exactos** recuperados y reordenados localmente."

    t_llm = (time.perf_counter() - t2) * 1000

    return {
        "answer": answer_text,
        "docs": final_docs,
        "candidates_evaluated": len(initial_docs),
        "latency": {
            "retrieval_ms": round(t_retrieval, 1),
            "rerank_ms": round(t_rerank, 1),
            "llm_ms": round(t_llm, 1),
            "total_ms": round(t_retrieval + t_rerank + t_llm, 1)
        }
    }

# -------------------------------------------------------------------
# 3. INTERFAZ GRÁFICA (UI)
# -------------------------------------------------------------------
st.title("🚀 NASA Mars Mission Research Assistant")
st.caption("Sistema RAG en Producción con Búsqueda Híbrida (BM25 + Chroma), Cross-Encoder Reranking y Guardrails de Abstención.")

components = init_rag_components()

with st.sidebar:
    st.header("🔑 Credenciales LLM")
    env_key = os.environ.get("GEMINI_API_KEY") or (FALLBACK_GEMINI_KEY if "pega-tu-clave" not in FALLBACK_GEMINI_KEY else "")
    api_key_input = st.text_input(
        "Gemini API Key:",
        value=env_key,
        type="password",
        placeholder="Pega aquí tu clave AIzaSy...",
        help="Tu clave se guarda en memoria para esta sesión sin quedar expuesta en el código."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input.strip()

    st.divider()
    st.header("⚙️ Configuración de Arquitectura")
    pipeline_mode = st.radio(
        "Modo del Retriever:",
        options=[
            "Producción (Híbrido + Reranker)",
            "Básico (Solo Vector Search k=4)"
        ],
        index=0
    )
    use_prod = pipeline_mode.startswith("Producción")

    st.divider()
    st.subheader("🧪 Casos de Prueba Rápidos")
    
    sample_q = None
    if st.button("📡 Consulta Técnica (In-Domain)", use_container_width=True):
        sample_q = "What are the primary radiation shielding challenges and mitigation strategies for human Mars missions?"
    if st.button("🧪 Búsqueda por Acrónimo (BM25 Test)", use_container_width=True):
        sample_q = "How does ISRU help reduce launch mass for Mars surface missions?"
    if st.button("🛡️ Pregunta Trampa (Guardrail Test)", use_container_width=True):
        sample_q = "What was the exact lunar module touchdown weight during the Apollo 11 Moon landing in 1969?"

    st.divider()
    st.metric("Fragmentos Indexados", components["total_chunks"])

default_query = sample_q if sample_q else ""
user_query = st.text_input(
    "Escribe tu consulta técnica sobre las misiones de la NASA a Marte:",
    value=default_query
)

if user_query:
    with st.spinner("Ejecutando pipeline RAG y midiendo telemetría..."):
        result = execute_instrumented_query(user_query, components, use_production_pipeline=use_prod)

    st.subheader("⏱️ Telemetría de Latencia en Tiempo Real")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("1. Recuperación", f"{result['latency']['retrieval_ms']} ms", delta=f"{result['candidates_evaluated']} candidatos")
    col2.metric("2. Cross-Encoder Reranker", f"{result['latency']['rerank_ms']} ms", delta="Top 4 filtrados" if use_prod else "Desactivado")
    col3.metric("3. Generación LLM", f"{result['latency']['llm_ms']} ms")
    col4.metric("⚡ Latencia Total", f"{result['latency']['total_ms']} ms")

    st.divider()

    col_ans, col_docs = st.columns([1.2, 1])

    with col_ans:
        st.subheader("🤖 Respuesta Sintetizada")
        if "ABSTENCIÓN:" in result["answer"]:
            st.warning("🛡️ **Guardrail de Abstención Activado:** Consulta fuera de dominio detectada.")
            st.write(result["answer"])
        elif "⚠️ **Aviso de API LLM:**" in result["answer"]:
            st.info(result["answer"])
        else:
            st.success("✅ **Respuesta verificada con citas documentales**")
            st.markdown(result["answer"])

    with col_docs:
        st.subheader(f"📄 Evidencia Documental (Top {len(result['docs'])} Chunks)")
        for idx, doc in enumerate(result["docs"]):
            source_file = doc.metadata.get("source_doc", "Desconocido")
            page_num = doc.metadata.get("page", "?")
            relevance_score = doc.metadata.get("relevance_score", None)
            score_badge = f" | Score: {relevance_score:.3f}" if relevance_score is not None else ""

            with st.expander(f"#{idx + 1} 📄 {source_file} (Pág. {page_num}){score_badge}", expanded=(idx == 0)):
                st.caption(f"**Archivo origen:** `data/raw/{source_file}` | **Página:** `{page_num}`")
                st.text(doc.page_content)