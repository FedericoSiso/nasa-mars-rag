import os
import sys
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR).lower() == "src" else CURRENT_DIR
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from ingestion import extract_and_chunk_pdfs, build_vector_store
from generator import SYSTEM_PROMPT, format_docs_with_metadata, get_llm, build_extractive_fallback
from retriever import BasicVectorRetriever, HybridRRFRetriever, ProductionRerankRetriever

# -------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# -------------------------------------------------------------------
st.set_page_config(
    page_title="NASA | Portal de Investigación y Telemetría NTRS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# SISTEMA DE DISEÑO NASA.GOV (CSS ALTO CONTRASTE & MINIMALISTA)
# -------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Fondo general negro espacial y tipografía blanca limpia */
    .stApp {
        background-color: #05070B;
        color: #FFFFFF;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Barra lateral estilo consola de misión */
    [data-testid="stSidebar"] {
        background-color: #090C14 !important;
        border-right: 1px solid #1A2030;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }

    /* Ocultar decoración superior por defecto de Streamlit */
    header[data-testid="stHeader"] {
        background: rgba(5, 7, 11, 0.85);
        backdrop-filter: blur(8px);
    }

    /* Barra de navegación superior estilo nasa.gov/es */
    .nasa-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.5rem;
        background-color: #000000;
        border-bottom: 1px solid #1E2638;
        margin-bottom: 1.5rem;
        border-radius: 6px;
    }
    .nasa-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .nasa-badge {
        background-color: #FC3D21;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.78rem;
        letter-spacing: 0.14em;
        padding: 4px 10px;
        border-radius: 3px;
        text-transform: uppercase;
    }
    .nasa-title-nav {
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.08em;
        color: #FFFFFF;
        text-transform: uppercase;
    }
    .nasa-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #F59E0B;
        letter-spacing: 0.06em;
    }

    /* Hero Banner con el Planeta Tierra (Home) */
    .earth-hero {
        position: relative;
        background: linear-gradient(90deg, rgba(5,7,11,0.94) 0%, rgba(5,7,11,0.72) 52%, rgba(5,7,11,0.25) 100%),
                    url('https://images-assets.nasa.gov/image/iss042e340858/iss042e340858~large.jpg');
        background-size: cover;
        background-position: center 35%;
        border: 1px solid #1E293B;
        border-left: 4px solid #FC3D21;
        border-radius: 8px;
        padding: 3.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.75);
    }
    .hero-kicker {
        color: #FC3D21;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-headline {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        color: #FFFFFF;
        margin-bottom: 0.85rem;
        letter-spacing: -0.02em;
    }
    .hero-subtext {
        font-size: 1.02rem;
        color: #CBD5E1;
        max-width: 680px;
        line-height: 1.6;
        font-weight: 300;
    }

    /* Tarjetas de Telemetría (KPIs alto contraste) */
    div[data-testid="stMetric"] {
        background-color: #0B0F19;
        border: 1px solid #1E293B;
        border-top: 3px solid #FC3D21;
        padding: 1.1rem 1.25rem;
        border-radius: 6px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #FFFFFF !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #F59E0B !important;
    }

    /* Tarjetas informativas del Home (Tierra -> Marte) */
    .mission-card {
        background-color: #0B0F19;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 1.4rem;
        height: 100%;
        transition: border-color 0.2s ease;
    }
    .mission-card:hover {
        border-color: #FC3D21;
    }
    .mission-card-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #F59E0B;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .mission-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .mission-card-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* Botones estilo NASA (Oscuros con borde fino y hover rojo/dorado) */
    .stButton > button {
        background-color: #0F1523 !important;
        color: #FFFFFF !important;
        border: 1px solid #263147 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.55rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #FC3D21 !important;
        border-color: #FC3D21 !important;
        color: #FFFFFF !important;
    }

    /* Campo de búsqueda principal */
    .stTextInput > div > div > input {
        background-color: #0B0F19 !important;
        color: #FFFFFF !important;
        border: 1px solid #2E3A52 !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FC3D21 !important;
        box-shadow: 0 0 0 1px #FC3D21 !important;
    }

    /* Contenedor de respuesta verificada */
    .nasa-response-box {
        background-color: #0B0F19;
        border: 1px solid #1E293B;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 1.5rem;
        line-height: 1.65;
        color: #F8FAFC;
    }
    .nasa-guardrail-box {
        background-color: #140A0A;
        border: 1px solid #7F1D1D;
        border-left: 4px solid #FC3D21;
        border-radius: 6px;
        padding: 1.5rem;
        color: #FCA5A5;
    }

    /* Expanders de evidencia documental */
    div[data-testid="stExpander"] {
        background-color: #0B0F19 !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        margin-bottom: 0.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. CARGA EN CACHÉ DE MOTORES RAG
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Sincronizando base vectorial NTRS, índice BM25 RRF y Cross-Encoder...")
def load_rag_engine_v2():
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
# 2. EJECUCIÓN CON TELEMETRÍA POR ETAPA
# -------------------------------------------------------------------
def execute_instrumented_query(query: str, components: dict, use_production_pipeline: bool = True):
    t0 = time.perf_counter()
    if use_production_pipeline:
        initial_docs = components["hybrid_retriever"].invoke(query)
    else:
        initial_docs = components["basic_retriever"].invoke(query)
    t_retrieval = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    if use_production_pipeline:
        final_docs = components["prod_reranker"].rerank(query, initial_docs)
    else:
        final_docs = initial_docs
    t_rerank = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    context_str = format_docs_with_metadata(final_docs)
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
            answer_text = build_extractive_fallback(query, final_docs)
    except Exception:
        answer_text = build_extractive_fallback(query, final_docs)

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

# Inicializar estado de navegación
if "active_query" not in st.session_state:
    st.session_state.active_query = ""

components = load_rag_engine_v2()

# -------------------------------------------------------------------
# 3. BARRA LATERAL (CONTROL DE MISIÓN)
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌍 NASA // NTRS LAB")
    st.caption("SISTEMA DE RECUPERACIÓN HÍBRIDA Y TELEMETRÍA")
    
    if st.button("🌍 Inicio (Órbita Terrestre)", use_container_width=True):
        st.session_state.active_query = ""
        st.rerun()

    st.divider()
    st.markdown("#### ⚙️ ARQUITECTURA RAG")
    pipeline_mode = st.radio(
        "Motor de Búsqueda:",
        options=[
            "Producción (BM25 + RRF + Reranker)",
            "Básico (Vector Search k=4)"
        ],
        index=0
    )
    use_prod = pipeline_mode.startswith("Producción")

    st.divider()
    st.markdown("#### 🧪 VECTORES DE PRUEBA")
    
    if st.button("📡 Blindaje contra Radiación", use_container_width=True):
        st.session_state.active_query = "What are the primary radiation shielding challenges and mitigation strategies for human Mars missions?"
    if st.button("⚡ Producción ISRU en Superficie", use_container_width=True):
        st.session_state.active_query = "How does ISRU help reduce launch mass for Mars surface missions?"
    if st.button("🛡️ Prueba Guardrail (Apolo 11)", use_container_width=True):
        st.session_state.active_query = "What was the exact lunar module touchdown weight during the Apollo 11 Moon landing in 1969?"

    st.divider()
    st.metric("FRAGMENTOS INDEXADOS", components["total_chunks"], delta="NTRS Verified")

# -------------------------------------------------------------------
# 4. CABECERA OFICIAL & HERO DEL PLANETA TIERRA (HOME)
# -------------------------------------------------------------------
st.markdown("""
<div class="nasa-navbar">
    <div class="nasa-brand">
        <span class="nasa-badge">NASA</span>
        <span class="nasa-title-nav">Centro de Investigación de Misiones // De la Tierra a Marte</span>
    </div>
    <div class="nasa-status">● TELEMETRÍA NTRS ACTIVA</div>
</div>

<div class="earth-hero">
    <div class="hero-kicker">ESTACIÓN BASE TIERRA // ARCHIVO TÉCNICO NTRS</div>
    <div class="hero-headline">Exploración Humana más allá de la Órbita Terrestre</div>
    <div class="hero-subtext">
        Plataforma de inteligencia documental impulsada por Búsqueda Híbrida (BM25 + ChromaDB), 
        Reordenamiento Cross-Encoder de precisión y Guardrails deterministas contra alucinaciones.
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 5. CONSOLA DE BÚSQUEDA
# -------------------------------------------------------------------
query_input = st.text_input(
    "🔎 CONSULTAR INFORMES TÉCNICOS DE LA NASA:",
    value=st.session_state.active_query,
    placeholder="Escribe una consulta aeroespacial (ej. sistemas de soporte vital ECLSS, propulsión, radiación o ISRU)..."
)

if query_input != st.session_state.active_query:
    st.session_state.active_query = query_input

# -------------------------------------------------------------------
# 6. VISTA HOME (PLANETA TIERRA) VS VISTA DE RESULTADOS Y TELEMETRÍA
# -------------------------------------------------------------------
if not st.session_state.active_query:
    st.markdown("#### 🛰️ ARQUITECTURA DE GRADO DE PRODUCCIÓN (BENCHMARK VERIFICADO)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="mission-card">
            <div class="mission-card-tag">ETAPA 01 // RECALL 80%</div>
            <div class="mission-card-title">Fusión Híbrida (BM25 + Vector RRF)</div>
            <div class="mission-card-desc">
                Combina búsqueda semántica densa (BAAI/bge-small) con coincidencia léxica exacta para capturar acrónimos aeroespaciales críticos como ISRU, MOXIE y ECLSS.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="mission-card">
            <div class="mission-card-tag">ETAPA 02 // PRECISION@1 90%</div>
            <div class="mission-card-title">Cross-Encoder Reranker (MRR 0.933)</div>
            <div class="mission-card-desc">
                Evalúa pares pregunta-documento mediante atención cruzada (ms-marco-MiniLM-L-6-v2), elevando la página exacta al puesto #1 frente al 10% del RAG básico.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="mission-card">
            <div class="mission-card-tag">ETAPA 03 // ABSTENCIÓN 100%</div>
            <div class="mission-card-title">Guardrail & Citas Estrictas</div>
            <div class="mission-card-desc">
                Fuerza el anclaje documental por archivo y página exacta, bloqueando consultas fuera de dominio sin generar alucinaciones.
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    with st.spinner("Procesando telemetría de recuperación y síntesis..."):
        result = execute_instrumented_query(
            st.session_state.active_query,
            components,
            use_production_pipeline=use_prod
        )

    st.markdown("#### ⏱️ TELEMETRÍA DE LATENCIA EN TIEMPO REAL")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "1. RECUPERACIÓN RRF",
        f"{result['latency']['retrieval_ms']} ms",
        delta=f"{result['candidates_evaluated']} candidatos"
    )
    col2.metric(
        "2. CROSS-ENCODER RERANK",
        f"{result['latency']['rerank_ms']} ms",
        delta="Top 4 filtrados" if use_prod else "Bypass activo"
    )
    col3.metric(
        "3. SÍNTESIS & GUARDRAIL",
        f"{result['latency']['llm_ms']} ms",
        delta="Temp 0.0"
    )
    col4.metric(
        "⚡ LATENCIA END-TO-END",
        f"{result['latency']['total_ms']} ms",
        delta="SLA Nominal"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_ans, col_docs = st.columns([1.15, 1])

    with col_ans:
        st.markdown("#### 📡 TRANSMISIÓN SINTETIZADA")
        if "ABSTENCIÓN:" in result["answer"]:
            st.markdown(f"""
            <div class="nasa-guardrail-box">
                <strong>🛡️ PROTOCOLO DE ABSTENCIÓN ACTIVADO</strong><br><br>
                {result["answer"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="nasa-response-box">
                {result["answer"]}
            </div>
            """, unsafe_allow_html=True)

    with col_docs:
        st.markdown(f"#### 📄 EVIDENCIA NTRS (TOP {len(result['docs'])} FRAGMENTOS)")
        for idx, doc in enumerate(result["docs"]):
            source_file = doc.metadata.get("source_doc", "Desconocido")
            page_num = doc.metadata.get("page", "?")
            relevance_score = doc.metadata.get("relevance_score", None)
            score_badge = f" | RERANK SCORE: {relevance_score:.3f}" if relevance_score is not None else ""

            with st.expander(f"#{idx + 1} — {source_file} (PÁG. {page_num}){score_badge}", expanded=(idx == 0)):
                st.caption(f"ORIGEN: `data/raw/{source_file}`  |  PÁGINA: `{page_num}`")
                st.code(doc.page_content, language="text")
